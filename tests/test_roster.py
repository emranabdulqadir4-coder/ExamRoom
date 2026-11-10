"""Roster loading, name suggestion logic, correction by teachers."""
import sqlite3
import time

import cv2
import numpy as np
import pytest

from examguard.config import Config
from examguard.identify import Identifier
from examguard.models import Event, FaceObs
from examguard.roster import Roster, RosterError, Student
from examguard.service import build_services
from examguard.store import Store
from examguard.web.app import create_app

COLORS = {"Ahmed": (0, 0, 200), "نورة": (0, 200, 0), "Omar": (200, 0, 0)}     # BGR


def photo(color, size=120):
    img = np.full((size, size, 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class ColorEmbedder:
    """Stand-in for SFace: a face is its mean colour. Lets us test the matching logic offline."""
    def embed(self, img):
        v = img.reshape(-1, 3).mean(axis=0) - 100.0
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 1 else None


# ------------------------------------------------------------------ loading
def test_csv_roster_with_arabic_names_and_photos_folder(tmp_path):
    (tmp_path / "photos").mkdir()
    for name, col in COLORS.items():
        (tmp_path / "photos" / f"{name}.jpg").write_bytes(photo(col))
    (tmp_path / "roster.csv").write_text(
        "Name,Student_ID,Photo\nAhmed,101,Ahmed.jpg\nنورة,102,نورة.jpg\nOmar,103,../../etc/passwd.jpg\nNoPhoto,104,\n",
        encoding="utf-8-sig")
    r = Roster.load(str(tmp_path / "roster.csv"))
    assert [s.name for s in r.students] == ["Ahmed", "نورة", "Omar", "NoPhoto"]
    assert r.get("101").photo_jpeg and r.get("102").photo_bgr is not None
    assert r.get("103").photo_bgr is None and r.get("104").photo_bgr is None     # traversal ignored
    assert len(r.warnings) == 2


def test_xlsx_roster_with_embedded_pictures(tmp_path):
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    (tmp_path / "ahmed.png").write_bytes(cv2.imencode(".png", np.full((90, 90, 3), COLORS["Ahmed"], np.uint8))[1].tobytes())
    (tmp_path / "omar.png").write_bytes(cv2.imencode(".png", np.full((90, 90, 3), COLORS["Omar"], np.uint8))[1].tobytes())
    wb = Workbook(); ws = wb.active
    ws.append(["Name", "Student ID", "Photo"])
    ws.append(["Ahmed", 101]); ws.append(["Omar", 103]); ws.append(["Nobody", 104])
    ws.add_image(XLImage(str(tmp_path / "ahmed.png")), "C2")
    ws.add_image(XLImage(str(tmp_path / "omar.png")), "C3")
    wb.save(tmp_path / "roster.xlsx")
    r = Roster.load(str(tmp_path / "roster.xlsx"))
    assert r.get("101").photo_bgr is not None and r.get("103").photo_bgr is not None
    assert r.get("104").photo_bgr is None
    assert tuple(r.get("101").photo_bgr[45, 45]) == COLORS["Ahmed"]            # right picture on the right row


def test_bad_rosters_give_clear_errors(tmp_path):
    with pytest.raises(RosterError):
        Roster.load(str(tmp_path / "missing.csv"))
    (tmp_path / "x.csv").write_text("foo,bar\n1,2\n")
    with pytest.raises(RosterError):
        Roster.load(str(tmp_path / "x.csv"))


# ---------------------------------------------------------------- matching
def make_identifier(**cfg):
    students = [Student(ref=str(i), name=n, student_id=str(i), photo_bgr=np.full((100, 100, 3), c, np.uint8))
                for i, (n, c) in enumerate(COLORS.items(), 1)]
    ident = Identifier(Config(**cfg), Roster(students), ColorEmbedder())
    assert ident.enroll() == []
    return ident


def frame_with(faces):
    img = np.full((300, 500, 3), 100, np.uint8)
    for f, name in faces:
        x, y, w, h = f.bbox
        img[y:y + h, x:x + w] = COLORS[name]
    return img


def face(tid, x, yaw=0.0):
    f = FaceObs(bbox=(x, 100, 80, 80), yaw=yaw)
    f.track_id = tid
    return f


def feed(ident, pairs, seconds=4.0, t0=0.0):
    t = t0
    while t < t0 + seconds:
        faces = [f for f, _ in pairs]
        ident.observe(t, frame_with(pairs), faces)
        t += 0.25


def test_names_are_suggested_only_after_several_samples():
    ident = make_identifier()
    pairs = [(face(1, 20), "نورة")]
    ident.observe(0.0, frame_with(pairs), [pairs[0][0]])
    assert ident.identity(1) is None                                           # one sample is not enough
    feed(ident, pairs, seconds=3.0, t0=1.0)
    got = ident.identity(1)
    assert got and got.name == "نورة" and got.score > 0.9


def test_each_face_gets_its_own_name_and_turned_faces_are_ignored():
    ident = make_identifier()
    pairs = [(face(1, 20), "Ahmed"), (face(2, 150), "Omar"), (face(3, 300, yaw=60), "نورة")]
    feed(ident, pairs, seconds=6.0)
    assert ident.identity(1).name == "Ahmed" and ident.identity(2).name == "Omar"
    assert ident.identity(3) is None                                           # looking away: no sample taken


def test_unknown_face_gets_no_name():
    ident = make_identifier()
    stranger = np.full((300, 500, 3), 100, np.uint8)
    f = face(1, 20)
    stranger[100:180, 20:100] = (200, 200, 0)                                  # colour not in the roster
    for i in range(20):
        ident.observe(i * 0.8, stranger, [f])
    assert ident.identity(1) is None


def test_one_student_is_never_suggested_for_two_faces():
    ident = make_identifier()
    pairs = [(face(1, 20), "Omar"), (face(2, 150), "Omar")]
    feed(ident, pairs, seconds=6.0)
    named = [ident.identity(t) for t in (1, 2) if ident.identity(t)]
    assert len(named) <= 1


# ---------------------------------------------------- store + web + pipeline
def test_old_database_is_upgraded(tmp_path):
    db = sqlite3.connect(tmp_path / "old.db")
    db.executescript("""CREATE TABLE incidents (id INTEGER PRIMARY KEY AUTOINCREMENT, created_ts REAL NOT NULL,
        room TEXT NOT NULL, track_id INTEGER NOT NULL, kind TEXT NOT NULL, severity TEXT NOT NULL, score REAL NOT NULL,
        details TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', reviewer TEXT, reviewed_ts REAL, note TEXT,
        clip_name TEXT, thumb_name TEXT, purged_ts REAL);
        INSERT INTO incidents (created_ts, room, track_id, kind, severity, score, details) VALUES (1,'R',2,'phone','high',.9,'d');""")
    db.commit(); db.close()
    s = Store(tmp_path / "old.db")
    inc = s.list()[0]
    assert inc["student"] == "Student 2" and inc["student_name"] is None


def test_teacher_can_correct_the_name_and_it_is_audited(tmp_path):
    cfg = Config(data_dir=str(tmp_path), password="pw")
    svc = build_services(cfg, demo=True, demo_auto=False)
    c = create_app(svc).test_client()
    c.post("/login", data={"name": "Ms A", "password": "pw"})
    assert len(c.get("/api/roster").json["roster"]) == 6
    assert c.get("/api/roster/demo2/photo").mimetype == "image/jpeg"
    ev = Event(ts=time.time(), track_id=2, kind="phone", severity="high", score=.9, details="d",
               student_name="Sara Alotaibi", student_ref="demo2", match_score=0.82)
    inc = svc.store.add_incident(ev, "R1", None, None)
    assert inc["student"] == "Sara Alotaibi" and "SUGGESTED" in inc["identity_note"]
    r = c.post(f"/api/incidents/{inc['id']}/student", json={"ref": "demo3"})
    assert r.json["student"] == "Omar Khan" and r.json["id_corrected"] and "teacher" in r.json["identity_note"]
    assert c.post(f"/api/incidents/{inc['id']}/student", json={"ref": "nobody"}).status_code == 400
    assert c.post("/api/incidents/999/student", json={"ref": ""}).status_code == 404
    actions = [a["detail"] for a in svc.store.list_audit() if a["action"] == "student_corrected"]
    assert actions == ["Sara Alotaibi -> Omar Khan"]
    csv_text = c.get("/api/export.csv").get_data(as_text=True)
    assert csv_text.startswith("﻿")
    svc.stop()


def test_demo_pipeline_attaches_names_to_incidents(tmp_path):
    cfg = Config(data_dir=str(tmp_path), password="pw", post_roll_seconds=1.0)
    svc = build_services(cfg, demo=True, demo_auto=False)
    svc.start()
    try:
        time.sleep(2.5)
        svc.world.trigger("phone", student=5)
        end = time.time() + 25
        while time.time() < end and not svc.store.list("pending"):
            time.sleep(0.2)
        inc = svc.store.list("pending")[0]
        assert inc["student"] == "Yusuf Demir" and inc["student_ref"] == "demo5" and inc["match_score"] > 0.8
    finally:
        svc.stop()


def test_email_mentions_that_the_name_is_only_suggested():
    from examguard.notifier import EmailNotifier
    n = EmailNotifier(Config(email_user="a@gmail.com", email_app_password="x", email_to="t@x.com"))
    inc = {"ts": time.time(), "room": "R1", "student": "نورة القحطاني", "kind_label": "Phone detected",
           "severity": "high", "details": "d", "identity_note": "Name SUGGESTED by face matching: check it against the video"}
    msg = n.build_message(inc)
    body = msg.get_content()
    assert "نورة القحطاني" in body and "SUGGESTED" in body
    msg.as_bytes()                                                               # Arabic must serialise cleanly
