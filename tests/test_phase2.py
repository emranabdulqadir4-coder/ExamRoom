"""Phase 2: performance profiles, seat map, optional second camera, system check, watchdog, self-test, launcher."""
import csv
import json
import sqlite3
import sys
import time
import types

import cv2
import numpy as np
import pytest

from examguard import profiles, selftest as st
from examguard.config import Config
from examguard.evaluation import load_labels
from examguard.healthcheck import run_checks
from examguard.models import Event, FaceObs
from examguard.seatmap import (Seat, SeatAssigner, SeatMap, SeatMapError, row_letters, suggest_from_faces,
                               validate_seats)
from examguard.service import build_services
from examguard.store import Store
from examguard.watchdog import Watchdog
from examguard.web.app import create_app


# ------------------------------------------------------------------ profiles
def test_laptop_is_the_default_and_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("EXAMGUARD_PROFILE", raising=False)
    cfg = Config.from_env(data_dir=str(tmp_path))
    assert cfg.profile == "laptop" and cfg.process_fps == Config().process_fps and cfg.max_width == Config().max_width


def test_profile_sets_defaults_but_explicit_settings_win(tmp_path, monkeypatch):
    monkeypatch.delenv("EXAMGUARD_PROFILE", raising=False)
    monkeypatch.delenv("EXAMGUARD_MAX_WIDTH", raising=False)
    cfg = Config.from_env(data_dir=str(tmp_path), profile="gpu", phone_enabled=False)
    assert cfg.profile == "gpu" and cfg.max_width == 1280 and cfg.face_detector == "staged" and cfg.device == "cuda"
    assert cfg.phone_enabled is False                                    # explicit flag beats the profile
    monkeypatch.setenv("EXAMGUARD_MAX_WIDTH", "800")
    assert Config.from_env(data_dir=str(tmp_path), profile="gpu").max_width == 800      # so does .env


def test_dashboard_choice_is_remembered_and_env_overrides_it(tmp_path, monkeypatch):
    monkeypatch.delenv("EXAMGUARD_PROFILE", raising=False)
    profiles.save_profile(tmp_path, "edge")
    assert Config.from_env(data_dir=str(tmp_path)).profile == "edge"
    monkeypatch.setenv("EXAMGUARD_PROFILE", "jetson")
    assert Config.from_env(data_dir=str(tmp_path)).profile == "jetson"


def test_unknown_profile_is_refused_and_only_laptop_claims_to_be_tested(tmp_path, monkeypatch):
    monkeypatch.delenv("EXAMGUARD_PROFILE", raising=False)
    with pytest.raises(ValueError):
        Config.from_env(data_dir=str(tmp_path), profile="quantum")
    with pytest.raises(ValueError):
        profiles.save_profile(tmp_path, "quantum")
    assert {p["name"]: p["tested"] for p in profiles.catalogue()} == \
        {"laptop": True, "edge": False, "jetson": False, "gpu": False}


def test_corrupt_settings_file_falls_back_to_laptop(tmp_path, monkeypatch):
    monkeypatch.delenv("EXAMGUARD_PROFILE", raising=False)
    (tmp_path / "settings.json").write_text("{not json")
    assert Config.from_env(data_dir=str(tmp_path)).profile == "laptop"


def test_cuda_request_without_a_gpu_falls_back_to_cpu():
    from examguard.vision import resolve_device
    assert resolve_device("auto") is None and resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") in ("cuda:0", "cpu")                   # never raises, never crashes without a GPU
    with pytest.raises(ValueError):
        resolve_device("tpu")


# ------------------------------------------------------------------ seat map
def face(tid, cx, cy, w=60):
    return FaceObs(bbox=(int(cx - w / 2), int(cy - w / 2), w, w), track_id=tid)


def test_row_letters():
    assert [row_letters(i) for i in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]


def test_validate_seats_rejects_bad_input():
    ok = {"label": "A1", "x": 0.5, "y": 0.5, "camera": 0}
    assert validate_seats([ok])[0].label == "A1"
    for bad in ({**ok, "label": ""}, {**ok, "label": "<b>"}, {**ok, "label": "x" * 30}, {**ok, "x": 1.5},
                {**ok, "y": float("nan")}, {**ok, "camera": 5}, {"label": "A1"}):
        with pytest.raises(SeatMapError):
            validate_seats([bad])
    with pytest.raises(SeatMapError):
        validate_seats([ok, {**ok, "label": "a1"}])                      # duplicate label, same camera
    validate_seats([ok, {**ok, "camera": 1}])                            # same label on the other camera is fine
    with pytest.raises(SeatMapError):
        validate_seats([ok] * 401)


def test_seatmap_roundtrip_and_corrupt_file(tmp_path):
    sm = SeatMap(tmp_path / "s.json")
    sm.replace([Seat("A1", 0.2, 0.3), Seat("B1", 0.7, 0.8, 1, "r9")])
    again = SeatMap(tmp_path / "s.json")
    assert [(s.label, s.camera, s.student_ref) for s in again.seats()] == [("A1", 0, ""), ("B1", 1, "r9")]
    assert len(again.seats(0)) == 1
    (tmp_path / "s.json").write_text("garbage")
    assert len(SeatMap(tmp_path / "s.json")) == 0                        # never crashes startup


def test_assigner_nearest_seat_one_face_per_seat_and_radius():
    sm = SeatMap()
    sm.replace([Seat("A1", 0.25, 0.5), Seat("A2", 0.75, 0.5)])
    a = SeatAssigner(sm, 0, radius_faces=1.6)
    got = a.assign([face(1, 250, 270), face(2, 760, 280)], 1000, 540)
    assert (got[1].label, got[2].label) == ("A1", "A2")
    assert a.assign([face(3, 500, 270)], 1000, 540) == {}                # between the seats: no guess
    two = a.assign([face(1, 250, 270), face(5, 262, 270)], 1000, 540)    # two faces on one seat: only one gets it
    assert len(two) == 1 and two[1].label == "A1"


def test_assigner_is_sticky_between_neighbouring_seats():
    sm = SeatMap()
    sm.replace([Seat("A1", 0.40, 0.5), Seat("A2", 0.50, 0.5)])
    a = SeatAssigner(sm, 0, radius_faces=1.6)
    assert a.assign([face(1, 410, 270, 100)], 1000, 540)[1].label == "A1"
    assert a.assign([face(1, 455, 270, 100)], 1000, 540)[1].label == "A1"   # nearly in the middle: keeps A1


def test_suggest_from_faces_groups_rows_and_numbers_left_to_right():
    faces = [(0.8, 0.21, 0.08, 0.12), (0.2, 0.20, 0.08, 0.12), (0.5, 0.19, 0.08, 0.12),
             (0.3, 0.62, 0.08, 0.12), (0.7, 0.60, 0.08, 0.12)]
    got = {s.label: s.x for s in suggest_from_faces(faces)}
    assert got == {"A1": 0.2, "A2": 0.5, "A3": 0.8, "B1": 0.3, "B2": 0.7}
    assert suggest_from_faces([]) == []


# ------------------------------------------------------- incidents carry seats
def test_store_keeps_seat_camera_and_explains_where_a_name_came_from(tmp_path):
    store = Store(tmp_path / "t.db")
    base = dict(ts=time.time(), track_id=3, kind="head_turn", severity="low", score=0.5, details="d")
    plain = store.add_incident(Event(**base), "R1", None, None)
    assert plain["seat"] is None and plain["student_label"] == "Student 3" and plain["camera"] == ""
    by_seat = store.add_incident(Event(**base, seat="B4", student_name="Sara", student_ref="r1", id_source="seat",
                                       camera="Back camera"), "R1", None, None)
    assert by_seat["seat"] == "B4" and by_seat["student_label"] == "Student 3 (Back camera)"
    assert "SEAT MAP" in by_seat["identity_note"] and "B4" in by_seat["identity_note"]
    clash = store.add_incident(Event(**base, seat="B4", student_name="Sara", student_ref="r1", id_source="face",
                                     seat_student="Omar"), "R1", None, None)
    assert "Omar" in clash["identity_note"] and "Check carefully" in clash["identity_note"]
    agree = store.add_incident(Event(**base, seat="B4", student_name="Sara", student_ref="r1", id_source="face",
                                     seat_student="Sara"), "R1", None, None)
    assert agree["identity_note"].startswith("Name SUGGESTED by face matching")
    fixed = store.set_student(clash["id"], "r2", "Omar", "T")
    assert fixed["identity_note"] == "Name chosen by a teacher"
    rows = list(csv.reader(store.export_confirmed_csv().lstrip("﻿").splitlines()))
    assert rows[0][-2:] == ["seat", "camera"]


def test_old_database_without_phase2_columns_is_upgraded(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE incidents (id INTEGER PRIMARY KEY AUTOINCREMENT, created_ts REAL NOT NULL,
        room TEXT NOT NULL, track_id INTEGER NOT NULL, kind TEXT NOT NULL, severity TEXT NOT NULL, score REAL NOT NULL,
        details TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', reviewer TEXT, reviewed_ts REAL, note TEXT,
        clip_name TEXT, thumb_name TEXT, purged_ts REAL);
        INSERT INTO incidents (created_ts, room, track_id, kind, severity, score, details) VALUES (1,'R',2,'phone','high',.9,'x');""")
    con.commit(); con.close()
    store = Store(db)
    old = store.list()[0]
    assert old["kind"] == "phone" and old["seat"] is None and old["camera"] == ""


class FakeRoster:
    def __init__(self, *students):
        self.s = {x.ref: x for x in students}

    def get(self, ref):
        return self.s.get(ref)


def _monitor_for_enrich(seatmap, roster, identifier=None):
    from examguard.monitor import Monitor
    m = Monitor.__new__(Monitor)
    m.camera_name, m.identifier, m.roster = "", identifier, roster
    return m


def test_seat_suggests_a_name_when_face_matching_has_none():
    sm = SeatMap(); sm.replace([Seat("B4", 0.5, 0.5, 0, "r1")])
    student = types.SimpleNamespace(ref="r1", name="Sara")
    m = _monitor_for_enrich(sm, FakeRoster(student))
    ev = Event(ts=1, track_id=7, kind="phone", severity="high", score=1, details="d")
    m._enrich(ev, {7: sm.seats()[0]})
    assert (ev.seat, ev.student_name, ev.student_ref, ev.id_source, ev.match_score) == ("B4", "Sara", "r1", "seat", None)


def test_face_match_wins_over_seat_but_a_disagreement_is_recorded():
    sm = SeatMap(); sm.replace([Seat("B4", 0.5, 0.5, 0, "r1")])
    sara, omar = (types.SimpleNamespace(ref="r1", name="Sara"), types.SimpleNamespace(ref="r2", name="Omar"))
    ident = types.SimpleNamespace(identity=lambda tid: types.SimpleNamespace(ref="r2", name="Omar", score=0.7))
    m = _monitor_for_enrich(sm, FakeRoster(sara, omar), ident)
    ev = Event(ts=1, track_id=7, kind="phone", severity="high", score=1, details="d")
    m._enrich(ev, {7: sm.seats()[0]})
    assert (ev.student_name, ev.id_source, ev.seat_student) == ("Omar", "face", "Sara")


def test_no_seat_no_roster_leaves_event_unchanged():
    m = _monitor_for_enrich(SeatMap(), None)
    ev = Event(ts=1, track_id=1, kind="phone", severity="high", score=1, details="d")
    m._enrich(ev, {})
    assert (ev.seat, ev.student_name, ev.id_source) == (None, None, None)


# ------------------------------------------- the whole thing, with two cameras
def wait_for(fn, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.2)
    return None


@pytest.fixture
def two_cam(tmp_path):
    cfg = Config(data_dir=str(tmp_path), password="pw", second_source="demo", post_roll_seconds=1.0)
    svc = build_services(cfg, demo=True, demo_auto=False)
    app = create_app(svc)
    svc.start()
    client = app.test_client()
    client.post("/login", data={"name": "T", "password": "pw"})
    yield svc, client
    svc.stop()


def test_second_camera_incidents_are_labelled_with_their_camera_and_seat(two_cam):
    svc, client = two_cam
    assert len(svc.all_monitors()) == 2 and svc.monitor is svc.monitors[0]
    time.sleep(4.5)                                    # every student's usual posture forms first (looking down is relative to it)
    svc.worlds[1].trigger("head_turn", student=2)
    svc.worlds[0].trigger("look_down", student=4)
    incs = wait_for(lambda: len(svc.store.list("pending")) >= 2 and svc.store.list("pending"))
    assert incs, "both cameras should have raised an incident"
    by_kind = {i["kind"]: i for i in incs}
    assert by_kind["looking_down"]["camera"] == "Front camera" and by_kind["looking_down"]["seat"] == "B1"
    assert by_kind["head_turn"]["camera"] == "Back camera" and by_kind["head_turn"]["seat"] == "C2"
    assert by_kind["head_turn"]["student_label"] == "Student 2 (Back camera)"
    st_ = client.get("/api/status").get_json()
    assert [c["camera"] for c in st_["cameras"]] == ["Front camera", "Back camera"]
    assert client.get("/video_feed?cam=9").status_code == 404
    assert client.get("/api/seatmap/snapshot?cam=1").status_code == 200
    r = client.post("/api/demo/trigger", json={"kind": "phone", "cam": 1, "student": 1})
    assert r.status_code == 200 and client.post("/api/demo/trigger", json={"kind": "phone", "cam": 7}).status_code == 400


def test_one_camera_room_has_no_camera_labels(tmp_path):
    svc = build_services(Config(data_dir=str(tmp_path), password="pw"), demo=True, demo_auto=False)
    assert len(svc.all_monitors()) == 1 and svc.monitor.camera_name == ""
    svc.store.close()


# ---------------------------------------------------------------- dashboard APIs
@pytest.fixture
def web(tmp_path):
    cfg = Config(data_dir=str(tmp_path), password="pw", post_roll_seconds=1.0)
    svc = build_services(cfg, demo=True, demo_auto=False)
    app = create_app(svc)
    svc.start()
    c = app.test_client()
    yield svc, c, tmp_path
    svc.stop()


def login(c):
    assert c.post("/login", data={"name": "Ms A", "password": "pw"}).status_code == 302


def test_new_pages_and_apis_need_sign_in(web):
    svc, c, _ = web
    for path in ("/seatmap", "/health"):
        assert c.get(path).status_code == 302
    for path in ("/api/seatmap", "/api/health", "/api/profile", "/api/seatmap/snapshot", "/api/seatmap/suggest"):
        assert c.get(path).status_code == 401
    assert c.put("/api/seatmap", json={"seats": []}).status_code == 401
    assert c.post("/api/profile", json={"profile": "gpu"}).status_code == 401
    assert c.post("/api/restart").status_code == 401


def test_seatmap_api_validation_and_audit(web):
    svc, c, _ = web
    login(c)
    seats = c.get("/api/seatmap").get_json()["seats"]
    assert len(seats) == 6
    for bad in ([{"label": "", "x": .5, "y": .5, "camera": 0}],
                [{"label": "Z1", "x": .5, "y": .5, "camera": 0, "student_ref": "nobody"}],
                [{"label": "Z1", "x": .5, "y": .5, "camera": 0, "student_ref": "demo1"},
                 {"label": "Z2", "x": .6, "y": .5, "camera": 0, "student_ref": "demo1"}]):
        assert c.put("/api/seatmap", json={"seats": bad}).status_code == 400
    ok = c.put("/api/seatmap", json={"seats": [{"label": "Q1", "x": .3, "y": .4, "camera": 0, "student_ref": "demo2"}]})
    assert ok.status_code == 200 and [s.label for s in svc.seatmap.seats()] == ["Q1"]
    assert any(a["action"] == "seatmap_saved" and a["actor"] == "Ms A" for a in svc.store.list_audit())
    assert wait_for(lambda: c.get("/api/seatmap/snapshot").status_code == 200, 10)     # needs a first frame
    assert c.get("/api/seatmap/snapshot").mimetype == "image/jpeg"
    assert wait_for(lambda: len(c.get("/api/seatmap/suggest").get_json()["seats"]) == 6, 10)


def test_profile_buttons_save_choice_and_restart_needs_the_launcher(web, monkeypatch):
    svc, c, tmp = web
    login(c)
    monkeypatch.delenv("EXAMGUARD_PROFILE", raising=False)
    monkeypatch.delenv("EXAMGUARD_SUPERVISED", raising=False)
    info = c.get("/api/profile").get_json()
    assert info["current"] == "laptop" and len(info["profiles"]) == 4 and info["supervised"] is False
    assert c.post("/api/profile", json={"profile": "nope"}).status_code == 400
    r = c.post("/api/profile", json={"profile": "edge"}).get_json()
    assert r["saved"] == "edge" and r["restart_needed"] is True
    assert json.loads((tmp / "settings.json").read_text())["profile"] == "edge"
    assert c.post("/api/restart").status_code == 409                       # not started with start.py


def test_health_page_is_green_for_the_demo_and_red_when_the_camera_is_lost(web, monkeypatch):
    svc, c, _ = web
    login(c)
    time.sleep(2.5)
    rep = c.get("/api/health").get_json()
    assert rep["overall"] in ("ok", "warn") and not [i for i in rep["items"] if i["level"] == "fail"]
    assert any(i["title"] == "Seat map" and i["level"] == "ok" for i in rep["items"])
    real = svc.monitor.status
    monkeypatch.setattr(svc.monitor, "status", lambda: {**real(), "state": "waiting for video", "source_state": "reconnecting"})
    rep = c.get("/api/health").get_json()
    assert rep["overall"] == "fail"
    assert any(i["level"] == "fail" and "video" in i["title"] for i in rep["items"])


def test_untested_profile_is_flagged_in_the_health_check(tmp_path):
    cfg = Config.from_env(data_dir=str(tmp_path), profile="edge")
    svc = build_services(cfg, demo=True, demo_auto=False)
    item = next(i for i in run_checks(svc)["items"] if i["title"] == "Performance profile")
    assert item["level"] == "warn" and "NOT been tested" in item["detail"]
    svc.store.close()


def test_crowded_camera_suggests_a_second_camera(tmp_path):
    cfg = Config(data_dir=str(tmp_path), max_students=6)
    svc = build_services(cfg, demo=True, demo_auto=False)
    svc.start()
    time.sleep(2.0)
    items = run_checks(svc)["items"]
    svc.stop()
    assert any("crowded" in i["title"] and "second camera" in i["advice"] for i in items)


# ------------------------------------------------------------------- watchdog
class FakeMon:
    def __init__(self, **st):
        self.st = {"camera": "", "source_state": "connected", "last_frame_age": 0.5, "uptime": 500, "state": "running", **st}

    def status(self):
        return self.st


def test_watchdog_only_trips_on_a_real_hang():
    clock = {"t": 1000.0}
    ok, hung = FakeMon(), FakeMon(last_frame_age=200)
    assert Watchdog(90, [ok], lambda r: None, clock=lambda: clock["t"]).check() is None
    assert "no new picture" in Watchdog(90, [ok, hung], lambda r: None).check()
    assert Watchdog(90, [FakeMon(source_state="reconnecting", last_frame_age=500)], lambda r: None).check() is None   # unplugged camera: not a hang
    assert "never produced" in Watchdog(90, [FakeMon(last_frame_age=None, uptime=300)], lambda r: None).check()
    assert Watchdog(90, [FakeMon(last_frame_age=None, uptime=10)], lambda r: None).check() is None
    bad = FakeMon(state="error", error="boom")
    w = Watchdog(90, [bad], lambda r: None, clock=lambda: clock["t"])
    assert w.check() is None
    clock["t"] += 100
    assert "failing" in w.check()
    bad.st["state"] = "running"
    assert w.check() is None


def test_watchdog_thread_calls_back_once():
    hits = []
    w = Watchdog(1, [FakeMon(last_frame_age=50)], hits.append, interval=0.05)
    w.start()
    assert wait_for(lambda: hits, 3)
    w.stop()
    assert len(hits) == 1


# ------------------------------------------------------------------- self-test
def test_selftest_timeline_labels_and_prompts():
    steps = st.build_script(repeats=2, phone=True, talking=True)
    slots, total = st.build_timeline(steps)
    assert len(steps) == 2 * 5 * 2
    for a, b in zip(slots, slots[1:]):
        assert a.end < a.next_at <= b.ready_at < b.start          # nothing overlaps; there is quiet time between actions
    labels = st.labels_for(slots)
    assert len(labels) == 10 and {k for _, _, k in labels} == {"head_turn", "looking_down", "eye_gaze", "phone", "talking"}
    assert len(slots) - len(labels) == 10                            # the decoys are never labelled
    first = slots[0]
    assert st.prompt_at(slots, total, 1)["phase"] == "lead_in"
    assert st.prompt_at(slots, total, first.ready_at + 0.1)["phase"] == "ready"
    assert st.prompt_at(slots, total, first.start + 0.1)["phase"] == "go"
    assert st.prompt_at(slots, total, first.end + 0.1)["phase"] == "rest"
    assert st.prompt_at(slots, total, total + 1)["phase"] == "done"
    default_kinds = {s.kind for s in st.build_script(1) if s.kind}
    assert default_kinds == {"head_turn", "looking_down", "eye_gaze"}


def test_selftest_labels_file_is_readable_by_the_evaluation_tool(tmp_path):
    slots, _ = st.build_timeline(st.build_script(2))
    out = tmp_path / "x.labels.csv"
    n = st.write_labels(slots, str(out))
    loaded = load_labels(str(out))
    assert n == len(loaded) == 6 and loaded[0].start == pytest.approx(slots[0].start, abs=0.01)


class FakeCamera:
    def __init__(self):
        self.i = 0

    def read(self, timeout=0.5):
        self.i += 1
        return np.full((240, 320, 3), self.i % 255, np.uint8), self.i


def test_recording_length_follows_the_clock_so_label_times_match_video_times(tmp_path):
    clock = {"t": 0.0}
    out = tmp_path / "rec.mp4"
    shown = []

    def sleep(s):
        clock["t"] += 0.04                                       # a slow camera: 25 loop passes per second

    length = st.record_session(FakeCamera(), str(out), [], total=6.0, fps=15.0, clock=lambda: clock["t"], sleep=sleep,
                               show=lambda fr, t: shown.append(t) or True)
    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert length == pytest.approx(6.0, abs=0.2) and abs(frames - 90) <= 3
    assert shown and shown[-1] < 6.0


def test_recording_can_be_stopped_early_and_reports_failure_without_a_camera(tmp_path):
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += 0.05
    length = st.record_session(FakeCamera(), str(tmp_path / "a.mp4"), [], total=60.0, fps=10.0,
                               clock=lambda: clock["t"], sleep=sleep, show=lambda fr, t: t < 2.0)
    assert 1.5 < length < 3.0

    class Dead:
        def read(self, timeout=0.5):
            clock["t"] += 5
            return None, -1
    with pytest.raises(RuntimeError, match="no picture"):
        st.record_session(Dead(), str(tmp_path / "b.mp4"), [], 5, clock=lambda: clock["t"], sleep=sleep)


def test_prompt_overlay_does_not_alter_the_original_frame():
    frame = np.zeros((300, 400, 3), np.uint8)
    drawn = st.draw_prompt(frame, st.prompt_at(*reversed(st.build_timeline(st.build_script(1))), 1.0) if False else
                           {"phase": "go", "title": "NOW", "text": "Turn", "left": 3, "n": 4, "i": 1})
    assert frame.sum() == 0 and drawn.sum() > 0


# ------------------------------------------------------------------- launcher
def run_launcher(monkeypatch, codes):
    import start
    calls, sleeps = [], []

    class P:
        def __init__(self, cmd, **kw):
            calls.append((cmd, kw))
            self.code = codes[len(calls) - 1]

        def wait(self, timeout=None):
            return self.code
    monkeypatch.setattr(start.subprocess, "Popen", P)
    monkeypatch.setattr(start.time, "sleep", sleeps.append)
    monkeypatch.setattr(start, "check_setup", lambda: True)
    monkeypatch.setattr(start, "ensure_env_file", lambda: None)
    return start, calls, sleeps


def test_launcher_restarts_on_75_and_on_crashes_but_not_on_a_normal_stop(monkeypatch):
    start, calls, _ = run_launcher(monkeypatch, [75, 1, 0])
    assert start.main(["--demo"]) == 0 and len(calls) == 3
    assert all(kw["env"]["EXAMGUARD_SUPERVISED"] == "1" for _, kw in calls)
    assert calls[0][0][-1] == "--demo"


def test_launcher_gives_up_on_a_crash_loop(monkeypatch):
    start, calls, _ = run_launcher(monkeypatch, [1] * 20)
    assert start.main(["--demo"]) == 1 and len(calls) == start.MAX_QUICK_CRASHES


def test_launcher_scripts_exist_and_use_the_launcher():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    assert "start.py" in (root / "start.sh").read_text() and "start.py" in (root / "start.bat").read_text()
    assert (root / "start.sh").read_text().startswith("#!/usr/bin/env bash")
