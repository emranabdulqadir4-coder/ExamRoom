"""Looking down, eye glance, talking and passing: simulated room -> real rules -> incident with evidence clip."""
import time

import pytest

from examguard.config import Config
from examguard.service import build_services


@pytest.fixture
def svc(tmp_path):
    s = build_services(Config(data_dir=str(tmp_path), password="pw", post_roll_seconds=1.0),
                       demo=True, demo_auto=False)
    s.start()
    yield s
    s.stop()


def wait_until(fn, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.25)
    return False


def found(svc):
    return {(i["kind"], i["student_label"]) for i in svc.store.list()}


def test_each_new_behaviour_is_flagged_on_the_right_student_and_nothing_else(svc):
    time.sleep(3.0)                                    # every student's usual posture forms first
    svc.world.trigger("look_down", student=1)
    svc.world.trigger("eye_gaze", student=4)
    svc.world.trigger("talking", student=2)            # students 2 and 3 are neighbours
    svc.world.trigger("passing", student=5)            # hand moves from student 5 to student 6
    expected = {("looking_down", "Student 1"), ("eye_gaze", "Student 4"),
                ("talking", "Student 2"), ("talking", "Student 3"), ("passing", "Student 5")}
    assert wait_until(lambda: expected <= found(svc)), f"missing: {expected - found(svc)}"
    time.sleep(2.0)
    assert found(svc) == expected, f"unexpected extra flags: {found(svc) - expected}"
    for inc in svc.store.list():
        assert inc["status"] == "pending" and inc["has_clip"] and inc["has_thumb"]
        assert inc["kind_label"] and inc["details"]
    passing = next(i for i in svc.store.list() if i["kind"] == "passing")
    assert "Student 6" in passing["details"] and passing["severity"] == "high"
    talk = next(i for i in svc.store.list() if i["kind"] == "talking" and i["student_label"] == "Student 2")
    assert "Student 3" in talk["details"]


def test_a_quiet_room_raises_no_false_alarms_for_the_new_rules(svc):
    time.sleep(16.0)
    assert svc.store.list() == []
    h = svc.monitor.status()
    assert h["hand_detector"] and h["phone_detector"] and h["students"] == 6
