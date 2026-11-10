"""Writing 'Fail' back to the roster spreadsheet when a teacher confirms."""
import csv
import os
import time

import cv2
import numpy as np
import pytest

from examguard.config import Config
from examguard.marks import MarkError, MarkWriter
from examguard.models import Event
from examguard.roster import Roster
from examguard.service import build_services
from examguard.web.app import create_app


def make_csv(tmp_path, rows=None, header="name,student_id,mark"):
    rows = rows if rows is not None else ["Ahmed,101,", "نورة,102,", "Omar,103,B+"]
    f = tmp_path / "roster.csv"
    f.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8-sig")
    return f


def read(f):
    return list(csv.reader(f.read_text(encoding="utf-8-sig").splitlines()))


def test_csv_set_and_clear_only_our_own_mark(tmp_path):
    f = make_csv(tmp_path)
    w = MarkWriter(Roster.load(str(f)))
    w.set("102")
    assert read(f) == [["name", "student_id", "mark"], ["Ahmed", "101", ""], ["نورة", "102", "Fail"], ["Omar", "103", "B+"]]
    assert w.clear("102") is not None and read(f)[2][2] == ""
    assert w.clear("101") is None                                  # nothing of ours there: untouched


def test_clear_never_erases_a_real_grade(tmp_path):
    f = make_csv(tmp_path)
    w = MarkWriter(Roster.load(str(f)))
    assert w.clear("103") is None and read(f)[3][2] == "B+"


def test_column_is_created_when_missing_and_backup_made_once(tmp_path):
    f = make_csv(tmp_path, rows=["Ahmed,101", "Omar,103"], header="name,student_id")
    w = MarkWriter(Roster.load(str(f)))
    w.set("101"); w.set("103")
    assert read(f) == [["name", "student_id", "Mark"], ["Ahmed", "101", "Fail"], ["Omar", "103", "Fail"]]
    assert len(list(tmp_path.glob("roster.csv.backup-*"))) == 1


def test_resorting_the_sheet_after_startup_cannot_mark_the_wrong_student(tmp_path):
    f = make_csv(tmp_path)
    w = MarkWriter(Roster.load(str(f)))
    rows = read(f)
    f.write_text("\n".join([",".join(rows[0])] + [",".join(r) for r in reversed(rows[1:])]) + "\n", encoding="utf-8-sig")
    w.set("101")
    assert [r for r in read(f) if r[0] == "Ahmed"][0][2] == "Fail"
    assert [r for r in read(f) if r[0] == "Omar"][0][2] == "B+"


def test_ambiguous_or_missing_row_is_refused(tmp_path):
    f = make_csv(tmp_path, rows=["Sam,,", "Sam,,", "Lena,,"])
    w = MarkWriter(Roster.load(str(f)))
    with pytest.raises(MarkError, match="exactly one row"):
        w.set("row2")                                              # two students called Sam, no id
    before = f.read_bytes()
    f.write_text("name,student_id,mark\nSomeone Else,9,\n", encoding="utf-8-sig")
    with pytest.raises(MarkError):
        w.set("row4")                                              # Lena is no longer in the sheet
    assert before != f.read_bytes()                                # (we replaced it ourselves; Lena not marked)
    assert "Fail" not in f.read_text(encoding="utf-8-sig")


def test_xlsx_numeric_ids_and_pictures_survive(tmp_path):
    from openpyxl import Workbook, load_workbook
    from openpyxl.drawing.image import Image as XLImage
    png = tmp_path / "a.png"
    png.write_bytes(cv2.imencode(".png", np.full((90, 90, 3), (0, 0, 200), np.uint8))[1].tobytes())
    wb = Workbook(); ws = wb.active
    ws.append(["Name", "Student ID", "Photo"])
    ws.append(["Ahmed", 20230101]); ws.append(["Omar", 20230103])
    ws.add_image(XLImage(str(png)), "C2")
    f = tmp_path / "roster.xlsx"; wb.save(f)
    w = MarkWriter(Roster.load(str(f)))
    w.set("20230103")
    out = load_workbook(f).worksheets[0]
    assert [c.value for c in out[1]] == ["Name", "Student ID", "Photo", "Mark"]
    assert out["D3"].value == "Fail" and out["D2"].value is None and out["B3"].value == 20230103
    assert len(out._images) == 1                                   # the photo is still there
    assert Roster.load(str(f)).get("20230101").photo_bgr is not None


def test_locked_file_raises_clear_error_and_leaves_sheet_unchanged(tmp_path, monkeypatch):
    f = make_csv(tmp_path)
    w = MarkWriter(Roster.load(str(f)))
    before = f.read_bytes()
    def deny(*a, **k): raise PermissionError("locked")
    monkeypatch.setattr(os, "replace", deny)
    with pytest.raises(MarkError, match="close it in Excel"):
        w.set("101")
    monkeypatch.undo()
    assert f.read_bytes() == before


# -------------------------------------------------------------- via the web app
@pytest.fixture
def web(tmp_path):
    svc = build_services(Config(data_dir=str(tmp_path), password="pw"), demo=True, demo_auto=False)
    c = create_app(svc).test_client()
    c.post("/login", data={"name": "Ms A", "password": "pw"})
    return svc, c, tmp_path / "demo-roster.csv"


def add(svc, ref="demo2", name="Sara Alotaibi"):
    ev = Event(ts=time.time(), track_id=2, kind="phone", severity="high", score=.9, details="d",
               student_name=name, student_ref=ref, match_score=.8)
    return svc.store.add_incident(ev, "R1", None, None)


def mark_of(f, name):
    return [r for r in read(f) if r[0] == name][0][2]


def test_confirm_writes_fail_dismiss_afterwards_removes_it(web):
    svc, c, f = web
    inc = add(svc)
    assert mark_of(f, "Sara Alotaibi") == ""
    r = c.post(f"/api/incidents/{inc['id']}/review", json={"decision": "confirm"})
    assert r.status_code == 200 and "Fail" in r.json["mark_note"]
    assert mark_of(f, "Sara Alotaibi") == "Fail" and mark_of(f, "Ahmed Alharbi") == ""
    r = c.post(f"/api/incidents/{inc['id']}/review", json={"decision": "dismiss", "note": "my mistake"})
    assert r.status_code == 200 and mark_of(f, "Sara Alotaibi") == ""
    acts = [a["action"] for a in svc.store.list_audit()]
    assert "mark_written" in acts and "mark_cleared" in acts
    svc.stop()


def test_dismiss_alone_never_touches_the_sheet(web):
    svc, c, f = web
    before = f.read_bytes()
    inc = add(svc)
    assert c.post(f"/api/incidents/{inc['id']}/review", json={"decision": "dismiss"}).status_code == 200
    assert f.read_bytes() == before
    svc.stop()


def test_unidentified_student_is_confirmed_but_nothing_is_written(web):
    svc, c, f = web
    before = f.read_bytes()
    inc = add(svc, ref=None, name=None)
    r = c.post(f"/api/incidents/{inc['id']}/review", json={"decision": "confirm"})
    assert r.status_code == 200 and r.json["status"] == "confirmed" and r.json["mark_note"] == ""
    assert f.read_bytes() == before
    svc.stop()


def test_failed_write_means_the_incident_is_not_confirmed(web, monkeypatch):
    svc, c, f = web
    inc = add(svc)
    def deny(*a, **k): raise PermissionError("locked")
    monkeypatch.setattr(os, "replace", deny)
    r = c.post(f"/api/incidents/{inc['id']}/review", json={"decision": "confirm"})
    assert r.status_code == 409 and "close it in Excel" in r.json["error"]
    assert svc.store.get(inc["id"])["status"] == "pending"
    svc.stop()


def test_correcting_the_name_after_confirming_moves_the_mark(web):
    svc, c, f = web
    inc = add(svc)
    c.post(f"/api/incidents/{inc['id']}/review", json={"decision": "confirm"})
    r = c.post(f"/api/incidents/{inc['id']}/student", json={"ref": "demo3"})
    assert r.status_code == 200
    assert mark_of(f, "Omar Khan") == "Fail" and mark_of(f, "Sara Alotaibi") == ""
    svc.stop()


def test_mark_is_kept_while_another_confirmed_incident_remains(web):
    svc, c, f = web
    a, b = add(svc), add(svc)
    for i in (a, b):
        c.post(f"/api/incidents/{i['id']}/review", json={"decision": "confirm"})
    c.post(f"/api/incidents/{a['id']}/review", json={"decision": "dismiss"})
    assert mark_of(f, "Sara Alotaibi") == "Fail"                    # incident b still stands
    c.post(f"/api/incidents/{b['id']}/review", json={"decision": "dismiss"})
    assert mark_of(f, "Sara Alotaibi") == ""
    svc.stop()
