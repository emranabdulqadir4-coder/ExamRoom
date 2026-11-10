"""Looking down, sideways eye glances, talking and passing: rule logic on synthetic observations."""
import math

import numpy as np
import pytest

from examguard.config import Config
from examguard.facefeatures import (eyes_down_score, gaze_ratio, mouth_openness, pitch_from_matrix)
from examguard.models import Detection, FaceObs
from examguard.rules import RuleEngine

FPS = 10.0


def face(tid, x=None, **kw):
    f = FaceObs(bbox=(x if x is not None else 200 * tid - 100, 100, 60, 60), **kw)
    f.track_id = tid
    return f


def run(eng, seconds, faces_fn, hands_fn=None, t0=0.0):
    events = []
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        hands = hands_fn(t) if hands_fn else []
        events += eng.update(t, faces_fn(t), hands)
    return events


def kinds(events):
    return [(e.kind, e.track_id) for e in events]


# ------------------------------------------------------------- feature maths
def synthetic_landmarks(iris_shift=0.0):
    pts = [(0.0, 0.0)] * 478
    pts = list(pts)
    pts[33], pts[133] = (100, 100), (140, 100)          # one eye, 40 px wide
    pts[263], pts[362] = (200, 100), (160, 100)         # other eye (corners listed the "wrong" way round)
    pts[468] = (120 + iris_shift * 40, 100)
    pts[473] = (180 + iris_shift * 40, 100)
    pts[13], pts[14], pts[78], pts[308] = (150, 200), (150, 210), (130, 205), (170, 205)
    return pts


@pytest.mark.parametrize("shift", [-0.2, 0.0, 0.15])
def test_gaze_ratio_follows_the_iris_and_is_positive_toward_image_right(shift):
    assert gaze_ratio(synthetic_landmarks(shift)) == pytest.approx(shift, abs=1e-6)


def test_gaze_and_mouth_are_none_without_enough_landmarks():
    assert gaze_ratio([(0, 0)] * 468) is None and mouth_openness([(0, 0)] * 10) is None


def test_mouth_openness_is_gap_over_width():
    assert mouth_openness(synthetic_landmarks()) == pytest.approx(10 / 40)


def rot_x(deg, scale=1.0):
    t = math.radians(deg)
    m = np.eye(4)
    m[:3, :3] = scale * np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    return m


@pytest.mark.parametrize("deg", [-35, 0, 12, 40])
def test_pitch_from_matrix_recovers_angle_and_sign_can_be_flipped(deg):
    assert pitch_from_matrix(rot_x(deg, scale=1.7)) == pytest.approx(deg, abs=1e-6)
    assert pitch_from_matrix(rot_x(deg), sign=-1.0) == pytest.approx(-deg, abs=1e-6)


def test_eyes_down_score_averages_both_eyes():
    assert eyes_down_score({"eyeLookDownLeft": 0.8, "eyeLookDownRight": 0.4}) == pytest.approx(0.6)
    assert eyes_down_score({"jawOpen": 1.0}) is None


# ------------------------------------------------------------- looking down
def test_head_dip_below_own_posture_fires_after_three_seconds():
    eng = RuleEngine(Config())
    run(eng, 4, lambda t: [face(1, pitch=-10.0)])
    ev = run(eng, 6, lambda t: [face(1, pitch=15.0)], t0=4.0)            # 25 degrees lower than usual
    assert kinds(ev) == [("looking_down", 1)] and "25" in ev[0].details
    assert 3.0 <= ev[0].ts - 4.0 <= 3.3


def test_short_dip_and_steady_writing_posture_do_not_fire():
    eng = RuleEngine(Config())
    assert run(eng, 30, lambda t: [face(1, pitch=22.0)]) == []           # always head-down writing: that is normal
    assert run(eng, 2, lambda t: [face(1, pitch=45.0)], t0=30.0) == []   # 2 s dip is shorter than 3 s
    assert run(eng, 4, lambda t: [face(1, pitch=22.0)], t0=32.0) == []


def test_eyes_down_alone_is_enough():
    eng = RuleEngine(Config())
    run(eng, 4, lambda t: [face(1, eyes_down=0.1)])
    ev = run(eng, 5, lambda t: [face(1, eyes_down=0.7)], t0=4.0)
    assert kinds(ev) == [("looking_down", 1)] and "Eyes" in ev[0].details


def test_repeated_dips_are_flagged_as_peeking():
    eng = RuleEngine(Config())
    run(eng, 4, lambda t: [face(1, pitch=0.0)])
    ev, t = [], 4.0
    for _ in range(4):
        ev += run(eng, 1.0, lambda _t: [face(1, pitch=30.0)], t0=t)
        ev += run(eng, 1.5, lambda _t: [face(1, pitch=0.0)], t0=t + 1.0)
        t += 2.5
    assert kinds(ev) == [("repeated_looking_down", 1)]


# ------------------------------------------------------------------ eye gaze
def test_sideways_eyes_with_head_forward_fires():
    eng = RuleEngine(Config())
    run(eng, 4, lambda t: [face(1, yaw=0.0, gaze_x=0.0)])
    ev = run(eng, 3, lambda t: [face(1, yaw=1.0, gaze_x=0.2)], t0=4.0)
    assert kinds(ev) == [("eye_gaze", 1)] and "right" in ev[0].details


def test_turned_head_is_left_to_the_head_turn_rule():
    eng = RuleEngine(Config())
    run(eng, 4, lambda t: [face(1, yaw=0.0, gaze_x=0.0)])
    ev = run(eng, 3, lambda t: [face(1, yaw=30.0, gaze_x=0.2)], t0=4.0)
    assert "eye_gaze" not in [k for k, _ in kinds(ev)]


def test_usual_off_centre_gaze_is_not_flagged():
    eng = RuleEngine(Config())
    assert run(eng, 20, lambda t: [face(1, yaw=0.0, gaze_x=0.18)]) == []


def test_repeated_sideways_glances():
    eng = RuleEngine(Config())
    run(eng, 4, lambda t: [face(1, yaw=0.0, gaze_x=0.0)])
    ev, t = [], 4.0
    for _ in range(5):
        ev += run(eng, 0.7, lambda _t: [face(1, yaw=0.0, gaze_x=0.25)], t0=t)
        ev += run(eng, 1.0, lambda _t: [face(1, yaw=0.0, gaze_x=0.0)], t0=t + 0.7)
        t += 1.7
    assert kinds(ev) == [("repeated_eye_gaze", 1)]


# ------------------------------------------------------------------- talking
def mouth(t, freq=2.5, amp=0.1):
    return 0.03 + amp * (0.5 + 0.5 * math.sin(2 * math.pi * freq * t))


def test_two_neighbours_talking_flags_both_and_names_the_other():
    eng = RuleEngine(Config())
    ev = run(eng, 6, lambda t: [face(1, mouth=mouth(t)), face(2, mouth=mouth(t + 0.1)), face(3, mouth=0.03)])
    assert sorted(kinds(ev)) == [("talking", 1), ("talking", 2)]
    assert "Student 2" in [e for e in ev if e.track_id == 1][0].details


def test_one_talker_alone_is_not_flagged_by_default_but_can_be():
    quiet = lambda t: [face(1, mouth=mouth(t)), face(2, mouth=0.03)]
    assert run(RuleEngine(Config()), 6, quiet) == []
    assert kinds(run(RuleEngine(Config(talk_min_students=1)), 6, quiet)) == [("talking", 1)]


def test_talkers_far_apart_are_not_a_group():
    far = lambda t: [face(1, x=100, mouth=mouth(t)), face(2, x=1200, mouth=mouth(t))]
    assert run(RuleEngine(Config()), 6, far) == []


def test_yawning_or_holding_the_mouth_open_is_not_talking():
    yawn = lambda t: [face(1, mouth=0.4), face(2, mouth=0.4)]
    assert run(RuleEngine(Config()), 8, yawn) == []


def test_brief_chat_shorter_than_three_seconds_is_ignored():
    def faces(t):
        m = mouth(t) if t < 2.0 else 0.03
        return [face(1, mouth=m), face(2, mouth=m)]
    assert run(RuleEngine(Config()), 8, faces) == []


# ------------------------------------------------------------------- passing
def hand_at(x, y=190):
    return [Detection("hand", 0.9, (int(x), int(y), 30, 30))]


def two_students(t):
    return [face(1, x=100), face(2, x=300)]


def test_hand_travelling_between_neighbours_is_flagged_for_the_giver():
    def hands(t):
        if t < 1.0:
            return hand_at(95)                                 # resting at student 1
        if t < 2.6:
            return hand_at(95 + (t - 1.0) / 1.6 * 210)         # crossing
        return hand_at(305)                                    # arrives at student 2
    ev = run(RuleEngine(Config()), 5, two_students, hands)
    assert kinds(ev) == [("passing", 1)] and "Student 2" in ev[0].details and ev[0].severity == "high"


def test_hand_that_stays_at_its_own_desk_or_only_flicks_is_ignored():
    eng = RuleEngine(Config())
    assert run(eng, 6, two_students, lambda t: hand_at(95 + 10 * math.sin(t))) == []
    flick = lambda t: hand_at(95) if t < 1.0 else (hand_at(305) if 1.0 <= t < 1.2 else hand_at(95))
    assert run(RuleEngine(Config()), 5, two_students, flick) == []


def test_hand_crossing_between_students_who_are_not_neighbours_is_ignored():
    far = lambda t: [face(1, x=100), face(2, x=1500)]
    def hands(t):
        return hand_at(95) if t < 1.5 else hand_at(1505)
    assert run(RuleEngine(Config()), 5, far, hands) == []


def test_each_rule_can_be_switched_off():
    cfg = Config(look_down_enabled=False, eye_gaze_enabled=False, talking_enabled=False, hands_enabled=False)
    eng = RuleEngine(cfg)
    run(eng, 4, lambda t: [face(1, pitch=0.0, gaze_x=0.0, mouth=0.03), face(2, mouth=0.03)])
    ev = run(eng, 6, lambda t: [face(1, pitch=40.0, gaze_x=0.3, yaw=0.0, mouth=mouth(t)), face(2, mouth=mouth(t))],
             hands_fn=lambda t: hand_at(95), t0=4.0)
    assert ev == []
