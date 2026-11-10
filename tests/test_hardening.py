"""HTTPS, face-size health warning, tuning report and roster photo checks."""
import stat
import threading
import time

import numpy as np
import pytest
import requests
from cryptography import x509

from examguard import tls
from examguard.config import Config
from examguard.health import face_health
from examguard.identify import check_roster
from examguard.models import FaceObs
from examguard.roster import Roster, Student
from examguard.service import build_services
from examguard.tuning import Collector, run
from examguard.web.app import create_app


# ------------------------------------------------------------------- HTTPS
def test_self_signed_certificate_is_reused_and_regenerated_when_address_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(tls, "primary_ip", lambda: "192.0.2.10")
    crt, key = tls.ensure_self_signed(tmp_path)
    first = crt.read_bytes()
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    san = x509.load_pem_x509_certificate(first).extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    assert {"127.0.0.1", "192.0.2.10"} <= {str(i) for i in san.get_values_for_type(x509.IPAddress)}
    tls.ensure_self_signed(tmp_path)
    assert crt.read_bytes() == first                                  # unchanged: reused
    monkeypatch.setattr(tls, "primary_ip", lambda: "192.0.2.77")      # DHCP gave the laptop a new address
    tls.ensure_self_signed(tmp_path)
    assert crt.read_bytes() != first
    assert "192.0.2.77" in {str(i) for i in x509.load_pem_x509_certificate(crt.read_bytes())
                            .extensions.get_extension_for_class(x509.SubjectAlternativeName).value
                            .get_values_for_type(x509.IPAddress)}


def test_dashboard_is_served_over_https_with_secure_cookie(tmp_path):
    from werkzeug.serving import make_server
    cfg = Config(data_dir=str(tmp_path), password="pw", https=True)
    svc = build_services(cfg, demo=True, demo_auto=False)
    app = create_app(svc)
    crt, key = tls.ensure_self_signed(tmp_path / "tls")
    srv = make_server("127.0.0.1", 0, app, threaded=True, ssl_context=(str(crt), str(key)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"https://127.0.0.1:{srv.server_port}"
        assert requests.get(base + "/healthz", verify=str(crt), timeout=5).json() == {"ok": True}
        r = requests.post(base + "/login", data={"name": "Ms A", "password": "pw"}, verify=str(crt),
                          allow_redirects=False, timeout=5)
        assert r.status_code == 302 and "Secure" in r.headers["Set-Cookie"]
        with pytest.raises(requests.exceptions.SSLError):             # unknown certificates are not trusted by default
            requests.get(base + "/healthz", timeout=5)
    finally:
        srv.shutdown(); svc.stop()


def test_http_mode_keeps_cookie_non_secure(tmp_path):
    svc = build_services(Config(data_dir=str(tmp_path), password="pw"), demo=True, demo_auto=False)
    app = create_app(svc)
    assert app.config["SESSION_COOKIE_SECURE"] is False
    svc.stop()


# --------------------------------------------------------------- face health
def test_face_health_flags_small_faces_with_advice():
    ok = face_health([80, 90, 120], 40)
    assert ok["small"] == 0 and ok["advice"] == "" and ok["median_px"] == 90
    bad = face_health([80, 25, 30], 40)
    assert bad["small"] == 2 and "2 of 3" in bad["advice"] and "MAX_WIDTH" in bad["advice"]
    assert face_health([], 40) == {"n": 0, "median_px": None, "small": 0, "advice": ""}


def test_status_reports_face_health_while_running(tmp_path):
    svc = build_services(Config(data_dir=str(tmp_path), password="pw"), demo=True, demo_auto=False)
    svc.start()
    try:
        time.sleep(2.0)
        h = svc.monitor.status()["face_health"]
        assert h["n"] == 6 and h["small"] == 0 and h["advice"] == ""
    finally:
        svc.stop()


# -------------------------------------------------------------------- tuning
def faces_at(yaws, width=80):
    out = []
    for i, y in enumerate(yaws, 1):
        f = FaceObs(bbox=(100 * i, 100, width, width), yaw=y); f.track_id = i
        out.append(f)
    return out


def test_tuning_warns_when_normal_movement_is_near_the_threshold():
    rng = np.random.default_rng(1)
    col = Collector()
    for _ in range(300):
        col.add(faces_at([rng.normal(0, 3), rng.choice([-30, 30]) + rng.normal(0, 2)]))   # student 2 fidgets a lot
    text = col.report(threshold=35)
    assert "WARNING: normal movement reaches" in text and "Raise EXAMGUARD_YAW_THRESHOLD_DEG" in text


def test_tuning_is_happy_with_calm_students_and_flags_tiny_faces():
    rng = np.random.default_rng(2)
    col = Collector()
    for _ in range(300):
        col.add(faces_at([rng.normal(5, 2), rng.normal(-4, 2)], width=30))
    text = col.report(threshold=35)
    assert "well below the threshold" in text and "under 40 px" in text


def test_tuning_with_no_faces_says_so():
    col = Collector()
    for _ in range(50):
        col.add([])
    assert "Not enough data" in col.report(35)


def test_tuning_runs_against_the_demo_room():
    from examguard.demo import DemoWorld
    world = DemoWorld(Config(), auto=False)
    col = run(world, world, 960, seconds=3.0)
    text = col.report(35)
    assert "Student 1" in text and "Student 6" in text and "well below" in text


# ------------------------------------------------------------- roster check
class VecEmbedder:
    """First pixel value picks a fixed unit vector; 7 means 'no face found'."""
    V = {1: [1.0, 0.0], 2: [0.96, 0.28], 3: [0.33, 0.944], 4: [0.0, -1.0]}

    def embed(self, img):
        v = int(img[0, 0, 0])
        if v == 7:
            return None
        e = np.array(self.V[v], np.float32)
        return e / np.linalg.norm(e)


def student(name, value=None):
    img = None if value is None else np.full((100, 100, 3), value, np.uint8)
    return Student(ref=name, name=name, photo_bgr=img)


def test_check_roster_reports_each_kind_of_problem():
    roster = Roster([student("Ahmed", 1), student("Copy of Ahmed", 2), student("Lookalike", 3),
                     student("Distinct", 4), student("Blurry", 7), student("NoPhoto")])
    text = "\n".join(check_roster(roster, VecEmbedder()))
    assert "Ahmed and Copy of Ahmed" in text and "SAME person" in text
    assert "Ahmed and Lookalike" in text and "look alike" in text
    assert "Blurry: no face found" in text and "NoPhoto: no usable photo" in text
    assert "Distinct" not in text


def test_check_roster_clean_roster_has_no_issues():
    assert check_roster(Roster([student("A", 1), student("B", 4)]), VecEmbedder()) == []


# ------------------------------------------------- tuning for the new signals
def test_decide_pitch_sign_covers_every_outcome():
    from examguard.tuning import decide_pitch_sign
    assert decide_pitch_sign(0.0, 25.0, 1.0)[0] == 1.0 and "Correct" in decide_pitch_sign(0.0, 25.0, 1.0)[1]
    sign, msg = decide_pitch_sign(0.0, -25.0, 1.0)
    assert sign == -1.0 and "EXAMGUARD_PITCH_SIGN=-1" in msg
    assert decide_pitch_sign(0.0, -25.0, -1.0)[0] == 1.0                   # already flipped once: flip back
    assert decide_pitch_sign(0.0, 2.0, 1.0)[0] is None and decide_pitch_sign(None, 5.0, 1.0)[0] is None


def test_pose_check_runs_both_phases_and_reports():
    from examguard.tuning import pose_check

    class Source:
        state = "playing"
        def read(self, timeout=0.5):
            time.sleep(0.01)
            return np.zeros((60, 80, 3), np.uint8), 1

    class Analyzer:                                   # an inverted camera: looking down lowers the reading
        pitch = 0.0
        def analyze(self, frame, ts):
            f = FaceObs(bbox=(10, 10, 50, 50), yaw=0.0, pitch=self.pitch)
            return [f]

    an, said = Analyzer(), []
    phases = iter([0.0, -30.0])
    msg = pose_check(Source(), an, Config(), say=said.append, wait=lambda prompt: setattr(an, "pitch", next(phases)), seconds=0.3)
    assert "EXAMGUARD_PITCH_SIGN=-1" in msg and said == [msg]


def test_tuning_report_covers_posture_gaze_and_mouth():
    rng = np.random.default_rng(3)
    col = Collector()
    for _ in range(300):
        f = FaceObs(bbox=(100, 100, 80, 80), yaw=rng.normal(0, 2), pitch=rng.normal(5, 12),
                    gaze_x=rng.normal(0, 0.01), mouth=0.03)
        f.track_id = 1
        col.add([f])
    text = col.report(35, 40, Config())
    assert "TOO CLOSE: raise EXAMGUARD_DOWN_THRESHOLD_DEG" in text           # restless posture is near the 20 degree limit
    assert "Eye position" in text and "OK for a quiet room" in text
