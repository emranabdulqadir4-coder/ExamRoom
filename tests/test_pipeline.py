"""End-to-end: simulated room -> real tracker/rules -> evidence clip -> encrypted vault -> database."""
import subprocess
import time

import pytest

from examguard.config import Config
from examguard.service import build_services


@pytest.fixture
def svc(tmp_path):
    cfg = Config(data_dir=str(tmp_path), process_fps=12, post_roll_seconds=1.0, pre_roll_seconds=3.5)
    s = build_services(cfg, demo=True, demo_auto=False)
    s.start()
    yield s
    s.stop()


def wait_for(fn, timeout=25.0):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.2)
    return None


def test_head_turn_produces_pending_incident_with_encrypted_playable_clip(svc, tmp_path):
    q = svc.hub.subscribe()
    time.sleep(2.5)                                    # let every student's baseline form
    svc.world.trigger("head_turn", student=2)
    incs = wait_for(lambda: svc.store.list("pending"))
    assert incs, "no incident was raised"
    inc = incs[0]
    assert inc["kind"] == "head_turn" and inc["student_label"] == "Student 2"
    assert inc["status"] == "pending"                  # the AI never decides anything by itself
    assert inc["has_clip"] and inc["has_thumb"]

    kind, payload = q.get(timeout=2)                   # and the dashboard hub was notified
    assert kind == "incident" and '"Student 2"' in payload

    names = svc.store.media_names(inc["id"])
    clip = svc.vault.get(names["clip"])
    assert clip[4:8] == b"ftyp"                        # a real MP4
    enc_files = list((tmp_path / "evidence").glob("*.enc"))
    assert enc_files and all(b"ftyp" not in f.read_bytes() for f in enc_files)   # encrypted at rest

    out = tmp_path / "probe.mp4"
    out.write_bytes(clip)
    probe = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-f", "null", "-"],
                           capture_output=True, text=True)
    assert probe.returncode == 0 and probe.stderr.strip() == ""
    assert svc.store.count("confirmed") == 0


def test_phone_incident_is_high_severity(svc):
    time.sleep(2.5)
    svc.world.trigger("phone", student=5)
    incs = wait_for(lambda: svc.store.list("pending"))
    assert incs and incs[0]["kind"] == "phone" and incs[0]["student_label"] == "Student 5"
    assert incs[0]["severity"] == "high"


def test_off_axis_student_and_idle_room_raise_nothing(svc):
    time.sleep(8.0)                                    # student 3 sits 24 deg off-axis the whole time
    assert svc.store.list() == []
    st = svc.monitor.status()
    assert st["state"] == "running" and st["students"] == 6
