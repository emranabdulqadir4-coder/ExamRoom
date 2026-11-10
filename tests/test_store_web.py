"""Review workflow, retention and dashboard security."""
import time

import pytest

from examguard.config import Config
from examguard.models import Event
from examguard.retention import apply_retention
from examguard.service import build_services
from examguard.store import ReviewError, Store
from examguard.vault import Vault, VaultError
from examguard.web.app import create_app


def ev(ts=None, tid=1, kind="head_turn"):
    return Event(ts=ts or time.time(), track_id=tid, kind=kind, severity="medium", score=0.7, details="d")


@pytest.fixture
def env(tmp_path):
    cfg = Config(data_dir=str(tmp_path), password="pw")
    return cfg, Store(tmp_path / "t.db"), Vault(tmp_path / "ev")


def add(cfg, store, vault, **kw):
    vault.put("clip-x.mp4", b"video"); vault.put("thumb-x.jpg", b"img")
    return store.add_incident(ev(**kw), "R1", "clip-x.mp4", "thumb-x.jpg")


def test_incidents_start_pending_and_review_needs_named_human(env):
    cfg, store, vault = env
    i = add(cfg, store, vault)
    assert i["status"] == "pending"
    with pytest.raises(ReviewError):
        store.review(i["id"], "confirm", "")
    with pytest.raises(ReviewError):
        store.review(i["id"], "banana", "Ms A")
    done = store.review(i["id"], "confirm", "Ms A", "saw it")
    assert (done["status"], done["reviewer"]) == ("confirmed", "Ms A")


def test_retention_deletes_dismissed_but_keeps_confirmed_and_young_pending(env):
    cfg, store, vault = env
    now = time.time()
    a = add(cfg, store, vault); store.review(a["id"], "dismiss", "T")
    vault.put("clip-y.mp4", b"v")
    b = store.add_incident(ev(), "R1", "clip-y.mp4", None); store.review(b["id"], "confirm", "T")
    vault.put("clip-z.mp4", b"v")
    c = store.add_incident(ev(), "R1", "clip-z.mp4", None)
    assert apply_retention(cfg, store, vault, now=now + 60) == 0
    assert apply_retention(cfg, store, vault, now=now + 2 * 3600) == 1          # dismissed after 1 h
    assert not vault.exists("clip-x.mp4") and vault.exists("clip-y.mp4") and vault.exists("clip-z.mp4")
    assert store.get(a["id"])["purged"] and store.get(a["id"])["status"] == "dismissed"
    apply_retention(cfg, store, vault, now=now + 15 * 86400)                     # pending expires at 14 d
    c2 = store.get(c["id"])
    assert c2["status"] == "expired" and not vault.exists("clip-z.mp4")
    with pytest.raises(ReviewError):
        store.review(c["id"], "confirm", "T")
    apply_retention(cfg, store, vault, now=now + 31 * 86400)                     # confirmed after 30 d
    assert not vault.exists("clip-y.mp4")


def test_vault_rejects_path_traversal_and_tampering(env):
    cfg, store, vault = env
    with pytest.raises(VaultError):
        vault.put("../evil", b"x")
    vault.put("a.bin", b"secret")
    (vault.root / "a.bin.enc").write_bytes(b"garbage")
    with pytest.raises(VaultError):
        vault.get("a.bin")


def test_web_requires_login_throttles_and_enforces_review_rules(tmp_path):
    cfg = Config(data_dir=str(tmp_path), password="pw")
    svc = build_services(cfg, demo=True, demo_auto=False)       # not started: no threads needed
    app = create_app(svc); c = app.test_client()
    assert c.get("/").status_code == 302
    assert c.get("/api/incidents").status_code == 401
    assert c.get("/media/1/clip").status_code == 401
    for _ in range(5):
        assert b"Wrong password" in c.post("/login", data={"name": "x", "password": "no"}).data
    assert b"Too many" in c.post("/login", data={"name": "x", "password": "pw"}).data   # locked out
    svc.stop()


def test_web_review_flow_and_demo_endpoint_disabled_outside_demo(tmp_path):
    cfg = Config(data_dir=str(tmp_path), password="pw")
    svc = build_services(cfg, demo=True, demo_auto=False)
    app = create_app(svc); c = app.test_client()
    assert c.post("/login", data={"name": "Ms A", "password": "pw"}).status_code == 302
    inc = svc.store.add_incident(ev(), "R1", None, None)
    r = c.post(f"/api/incidents/{inc['id']}/review", json={"decision": "dismiss", "note": "ok"})
    assert r.status_code == 200 and r.json["reviewer"] == "Ms A"
    assert c.post("/api/incidents/999/review", json={"decision": "dismiss"}).status_code == 404
    assert c.post(f"/api/incidents/{inc['id']}/review", json={"decision": "x"}).status_code == 409
    assert c.get("/media/%d/clip" % inc["id"]).status_code == 404       # no clip recorded
    assert "Content-Security-Policy" in c.get("/api/status").headers
    svc.world = None                                                    # pretend: not demo mode
    assert c.post("/api/demo/trigger", json={"kind": "phone"}).status_code == 404
    svc.stop()


def test_email_notifier_builds_text_only_message_and_is_off_without_settings(tmp_path):
    from examguard.notifier import EmailNotifier
    assert not EmailNotifier(Config()).enabled
    cfg = Config(email_user="a@gmail.com", email_app_password="abcd efgh ijkl mnop",
                 email_to="t1@x.com, t2@x.com", public_url="http://lan:5000")
    n = EmailNotifier(cfg)
    assert n.enabled and n.password == "abcdefghijklmnop" and n.recipients == ["t1@x.com", "t2@x.com"]
    inc = {"ts": time.time(), "room": "R1", "student": "Student 2", "kind_label": "Phone detected",
           "severity": "high", "details": "Phone visible"}
    msg = n.build_message(inc)
    assert "Student 2" in msg["Subject"] and msg["To"] == "t1@x.com, t2@x.com"
    assert "http://lan:5000" in msg.get_content() and not list(msg.iter_attachments())
    assert len(list(n.build_message(inc, b"jpegbytes").iter_attachments())) == 1
