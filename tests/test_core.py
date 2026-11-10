"""Unit tests for head-pose maths, tracker and rule engine (no camera needed)."""
import math

import pytest

from examguard.config import Config
from examguard.headpose import yaw_from_points
from examguard.models import Detection, FaceObs
from examguard.rules import RuleEngine
from examguard.tracker import CentroidTracker


# ---------------------------------------------------------------- head pose
def _project(yaw_deg, roll_deg=0.0):
    """Weak-perspective projection of a simple 3D face (nose, two eye corners)."""
    th, rl = math.radians(yaw_deg), math.radians(roll_deg)
    pts = {"nose": (0, 0, 0), "a": (-225, -170, 135), "b": (225, -170, 135)}
    out = {}
    for k, (x, y, z) in pts.items():
        xr = x * math.cos(th) + z * math.sin(th)       # yaw about the vertical axis
        yr = y
        xi = xr * math.cos(rl) - yr * math.sin(rl)     # roll in the image plane
        yi = xr * math.sin(rl) + yr * math.cos(rl)
        out[k] = (300 + xi, 200 + yi)
    return out


@pytest.mark.parametrize("yaw", [0, 10, 25, 40, 55, 70])
def test_yaw_recovered(yaw):
    p = _project(yaw)
    est = yaw_from_points(p["nose"], p["a"], p["b"])
    assert abs(abs(est) - yaw) < 1.0


def test_yaw_sign_flips_with_direction():
    p, q = _project(40), _project(-40)
    assert yaw_from_points(p["nose"], p["a"], p["b"]) * yaw_from_points(q["nose"], q["a"], q["b"]) < 0


def test_yaw_ignores_roll_and_eye_order():
    p = _project(30, roll_deg=15)
    a = yaw_from_points(p["nose"], p["a"], p["b"])
    b = yaw_from_points(p["nose"], p["b"], p["a"])
    assert abs(a - b) < 1e-6 and abs(abs(a) - 30) < 1.0


def test_yaw_degenerate():
    assert yaw_from_points((0, 0), (5, 5), (5, 5)) is None


# ------------------------------------------------------------------ tracker
def _face(x, y=100, w=60, yaw=0.0):
    return FaceObs(bbox=(x, y, w, w), yaw=yaw)


def test_tracker_stable_ids_and_new_faces():
    tr = CentroidTracker()
    f1 = tr.update(0.0, [_face(100), _face(300)])
    assert [f.track_id for f in f1] == [1, 2]
    f2 = tr.update(0.1, [_face(305), _face(104)])      # order swapped, small movement
    assert {f.bbox[0]: f.track_id for f in f2} == {305: 2, 104: 1}
    f3 = tr.update(0.2, [_face(104), _face(305), _face(600)])
    assert sorted(f.track_id for f in f3) == [1, 2, 3]


def test_tracker_survives_short_gap_but_not_long_one():
    tr = CentroidTracker(max_missing_seconds=2.0)
    tr.update(0.0, [_face(100)])
    tr.update(1.0, [])
    assert tr.update(1.5, [_face(110)])[0].track_id == 1
    tr.update(2.0, [])
    assert tr.update(5.0, [_face(110)])[0].track_id == 2


# -------------------------------------------------------------------- rules
FPS = 10.0


def run(engine, seconds, yaw_fn, faces_fn=None, objects_fn=None, t0=0.0, tracks=(1,)):
    """Feed the engine at FPS for `seconds`; yaw_fn(t) -> yaw (None = face not visible)."""
    events = []
    steps = int(seconds * FPS)
    for i in range(steps):
        t = t0 + i / FPS
        faces = []
        for tid in tracks:
            y = yaw_fn(t, tid) if yaw_fn.__code__.co_argcount == 2 else yaw_fn(t)
            if y is not None:
                f = FaceObs(bbox=(100 * tid, 100, 60, 60), yaw=y)
                f.track_id = tid
                faces.append(f)
        objs = objects_fn(t) if objects_fn else []
        events += engine.update(t, faces, objs)
    return events


@pytest.fixture
def cfg():
    return Config(yaw_threshold_deg=35, turn_seconds=2.0, cooldown_seconds=20.0)


def test_sustained_turn_fires_once_after_threshold_time(cfg):
    eng = RuleEngine(cfg)
    run(eng, 3, lambda t: 2.0)                                   # calibrate
    ev = run(eng, 6, lambda t: 50.0, t0=3.0)
    assert len(ev) == 1 and ev[0].kind == "head_turn"
    assert 2.0 <= ev[0].ts - 3.0 <= 2.3
    assert ev[0].bbox == (100, 100, 60, 60) and ev[0].severity == "medium"


def test_short_turn_does_not_fire(cfg):
    eng = RuleEngine(cfg)
    run(eng, 3, lambda t: 0.0)
    ev = run(eng, 1.5, lambda t: 50.0, t0=3.0) + run(eng, 3, lambda t: 0.0, t0=4.5)
    assert ev == []


def test_per_student_baseline_handles_off_axis_seat(cfg):
    """A student seated at the edge of the frame who always looks 30 deg off-axis must not alert."""
    eng = RuleEngine(cfg)
    assert run(eng, 20, lambda t: 30.0) == []
    ev = run(eng, 5, lambda t: 70.0, t0=20.0)                    # a further 40 deg turn is real
    assert [e.kind for e in ev] == ["head_turn"]


def test_without_baseline_off_axis_student_would_alert():
    eng = RuleEngine(Config(use_baseline=False, yaw_threshold_deg=35, turn_seconds=2.0))
    assert [e.kind for e in run(eng, 5, lambda t: 40.0)] == ["head_turn"]


def test_face_lost_while_turned_still_counts(cfg):
    eng = RuleEngine(Config(turn_seconds=2.0, turn_gap_seconds=1.5))
    run(eng, 3, lambda t: 0.0)
    ev = run(eng, 1.5, lambda t: 50.0, t0=3.0)                  # turning, still visible
    ev += run(eng, 1.4, lambda t: None, t0=4.5)                 # then the landmarks vanish
    assert [e.kind for e in ev] == ["head_turn"]


def test_repeated_glancing(cfg):
    eng = RuleEngine(Config(glance_count=4, glance_window_seconds=60, turn_seconds=2.0))
    run(eng, 3, lambda t: 0.0)
    ev = []
    t = 3.0
    for _ in range(4):
        ev += run(eng, 1.0, lambda _t: 50.0, t0=t)               # 1 s glance
        ev += run(eng, 1.0, lambda _t: 0.0, t0=t + 1.0)
        t += 2.0
    assert [e.kind for e in ev] == ["repeated_glancing"]


def test_cooldown_suppresses_repeat_alerts():
    eng = RuleEngine(Config(turn_seconds=1.0, cooldown_seconds=20.0))
    run(eng, 3, lambda t: 0.0)
    ev = run(eng, 4, lambda t: 60.0, t0=3.0) + run(eng, 3, lambda t: 0.0, t0=7.0)
    ev += run(eng, 4, lambda t: 60.0, t0=10.0)                   # second turn only 7 s later
    assert len(ev) == 1


def test_phone_assigned_to_nearest_student_and_needs_duration(cfg):
    eng = RuleEngine(Config(phone_seconds=1.0))
    phone = lambda t: [Detection("cell phone", 0.8, (212, 170, 25, 40))]   # below student 2's face
    ev = run(eng, 0.8, lambda t: 0.0, objects_fn=phone, tracks=(1, 2, 3))
    assert ev == []
    ev = run(eng, 2.0, lambda t: 0.0, objects_fn=phone, tracks=(1, 2, 3), t0=0.8)
    assert [(e.kind, e.track_id, e.severity) for e in ev] == [("phone", 2, "high")]


def test_phone_far_from_every_face_is_ignored(cfg):
    eng = RuleEngine(cfg)
    phone = lambda t: [Detection("cell phone", 0.9, (900, 600, 25, 40))]
    assert run(eng, 3, lambda t: 0.0, objects_fn=phone, tracks=(1, 2)) == []


def test_non_phone_objects_ignored(cfg):
    eng = RuleEngine(cfg)
    thing = lambda t: [Detection("book", 0.9, (100, 160, 25, 40))]
    assert run(eng, 3, lambda t: 0.0, objects_fn=thing) == []
