"""Phase 2: learning from teacher decisions (dismiss reasons, Ignore, exams, missed reports, report, suggestions)."""
import json
import sqlite3
import time

import pytest

from examguard import learning
from examguard.config import Config
from examguard.models import Event
from examguard.service import build_services
from examguard.store import ReviewError, Store
from examguard.web.app import create_app


def ev(kind="head_turn", ts=None, tid=1):
    return Event(ts=ts or time.time(), track_id=tid, kind=kind, severity="medium", score=0.7, details="d")


@pytest.fixture
def env(tmp_path):
    cfg = Config(data_dir=str(tmp_path), password="pw")
    return cfg, Store(tmp_path / "t.db")


def decide(store, cfg, n, decision, reason=None, who="Ms A", kind="head_turn"):
    out = []
    for _ in range(n):
        i = store.add_incident(ev(kind=kind), cfg.room_name, None, None)
        out.append(store.review(i["id"], decision, who, reason=reason))
    return out


# ------------------------------------------------------------------ 1. reasons
def test_dismiss_reason_is_stored_validated_and_audited(env):
    cfg, store = env
    i = store.add_incident(ev(), cfg.room_name, None, None)
    with pytest.raises(ReviewError):
        store.review(i["id"], "dismiss", "Ms A", reason="because")
    with pytest.raises(ReviewError):
        store.review(i["id"], "confirm", "Ms A", reason="clock")          # only a dismissal has a reason
    done = store.review(i["id"], "dismiss", "Ms A", reason="clock")
    assert done["dismiss_reason"] == "clock" and done["dismiss_reason_label"] == "Looked at the clock"
    assert any("reason: clock" in a["detail"] for a in store.list_audit())
    # withdrawing the dismissal into a confirmation drops the reason (the audit trail keeps the history)
    assert store.review(i["id"], "confirm", "Ms A")["dismiss_reason"] is None


def test_reason_is_optional_in_the_store_for_old_callers(env):
    cfg, store = env
    i = store.add_incident(ev(), cfg.room_name, None, None)
    assert store.review(i["id"], "dismiss", "Ms A")["dismiss_reason"] is None


def test_reason_can_be_corrected_later_and_only_on_dismissed(env):
    cfg, store = env
    a, = decide(store, cfg, 1, "dismiss", "clock")
    out = store.set_dismiss_reason(a["id"], "thinking", "Mr B")
    assert out["dismiss_reason"] == "thinking"
    assert any(x["action"] == "reason_corrected" and x["actor"] == "Mr B" and "clock -> thinking" in x["detail"] for x in store.list_audit())
    with pytest.raises(ReviewError):
        store.set_dismiss_reason(a["id"], "nonsense", "Mr B")
    c, = decide(store, cfg, 1, "confirm")
    with pytest.raises(ReviewError):
        store.set_dismiss_reason(c["id"], "clock", "Mr B")


# ------------------------------------------------------------------- 2. ignore
def test_ignore_keeps_the_alert_pending_goes_last_and_is_never_a_false_alarm(env):
    cfg, store = env
    first = store.add_incident(ev(ts=1000.0), cfg.room_name, None, None)
    second = store.add_incident(ev(ts=1001.0), cfg.room_name, None, None)
    out = store.ignore(first["id"], "Ms A")
    assert out["status"] == "pending" and out["ignored"] and out["ignored_by"] == "Ms A"
    assert [x["id"] for x in store.list("pending")] == [second["id"], first["id"]]      # ignored goes to the bottom
    rep = learning.build_report(store, room=cfg.room_name)
    g = rep["rules"][0]
    assert g["dismissed"] == 0 and g["not_decided"] == 2 and g["dismissed_share"] is None
    assert store.ignore(first["id"], "Ms A", on=False)["ignored"] is False
    store.ignore(first["id"], "Ms A")
    done = store.review(first["id"], "dismiss", "Ms A", reason="clock")                 # deciding clears the flag
    assert done["ignored"] is False
    with pytest.raises(ReviewError):
        store.ignore(first["id"], "Ms A")                                               # already decided
    with pytest.raises(ReviewError):
        store.ignore(second["id"], "")


# --------------------------------------------------------------------- 3. exams
def test_exam_starts_with_first_alert_finishes_and_summarises(env):
    cfg, store = env
    assert store.current_exam(cfg.room_name) is None
    decide(store, cfg, 2, "dismiss", "clock")
    decide(store, cfg, 1, "confirm")
    p = store.add_incident(ev(), cfg.room_name, None, None)
    store.ignore(p["id"], "Ms A")
    store.add_missed(cfg.room_name, "phone", "Ms A", "back row", 5)
    cur = store.current_exam(cfg.room_name)
    assert cur["exam"]["open"] and cur["counts"]["alerts"] == 4
    done = store.finish_exam(cfg.room_name, "Ms A")
    assert not done["exam"]["open"] and done["exam"]["ended_by"] == "Ms A"
    c = done["counts"]
    assert (c["confirmed"], c["dismissed"], c["waiting"], c["ignored"], c["missed"]) == (1, 2, 1, 1, 1)
    assert done["reasons"] == {"clock": 2}
    assert done["missed"][0]["kind_label"] == "Phone detected" and done["missed"][0]["note"] == "back row"
    with pytest.raises(ReviewError):
        store.finish_exam(cfg.room_name, "Ms A")                                         # nothing running
    nxt = store.add_incident(ev(), cfg.room_name, None, None)                           # the next alert = a new exam
    assert nxt["exam_id"] != done["exam"]["id"]
    assert len(store.list_exams(cfg.room_name)) == 2


def test_long_silence_ends_the_old_exam(env):
    cfg, store = env
    old = time.time() - 8 * 3600
    a = store.add_incident(ev(ts=old), cfg.room_name, None, None)
    b = store.add_incident(ev(ts=time.time()), cfg.room_name, None, None)
    assert a["exam_id"] != b["exam_id"]
    ended = store.exam_summary(a["exam_id"])["exam"]
    assert not ended["open"] and "silence" in ended["ended_by"]


def test_explicit_start_and_unknown_exam(env):
    cfg, store = env
    s = store.start_exam(cfg.room_name, "Ms A")
    assert s["exam"]["started_by"] == "Ms A"
    with pytest.raises(ReviewError):
        store.start_exam(cfg.room_name, "Ms A")
    assert store.exam_summary(999) is None


# ----------------------------------------------------------------- 4. missed
def test_missed_report_validation(env):
    cfg, store = env
    with pytest.raises(ReviewError):
        store.add_missed(cfg.room_name, "teleporting", "Ms A")
    with pytest.raises(ReviewError):
        store.add_missed(cfg.room_name, "phone", "Ms A", minutes_ago=500)
    with pytest.raises(ReviewError):
        store.add_missed(cfg.room_name, "phone", "", minutes_ago=1)
    ok = store.add_missed(cfg.room_name, "other", "Ms A", "x" * 500, 2)
    assert ok["kind"] == "other"
    assert len(store.exam_summary(ok["exam_id"])["missed"][0]["note"]) == 200


# ----------------------------------------------------------------- 5. report
def test_report_counts_by_rule_and_teacher_and_holds_no_names(env):
    cfg, store = env
    decide(store, cfg, 3, "dismiss", "clock", "Ms A")
    decide(store, cfg, 1, "dismiss", "bad_detection", "Ms A")
    decide(store, cfg, 1, "dismiss", None, "Mr B")
    decide(store, cfg, 2, "confirm", "", "Mr B")
    store.add_incident(ev(), cfg.room_name, None, None)                                 # never reviewed
    store.add_missed(cfg.room_name, "head_turn", "Ms A")
    store.add_missed(cfg.room_name, "other", "Ms A")
    rep = learning.build_report(store, room=cfg.room_name)
    g = next(x for x in rep["rules"] if x["kind"] == "head_turn")
    assert (g["alerts"], g["confirmed"], g["dismissed"], g["decided"]) == (8, 2, 5, 7)
    assert (g["normal_behaviour"], g["bad_detection"], g["no_reason"], g["not_decided"], g["missed"]) == (3, 1, 1, 1, 1)
    assert g["dismissed_share"] == round(5 / 7, 3) and g["normal_share"] == round(3 / 7, 3)
    t = {x["reviewer"]: x for x in rep["teachers"]}
    assert t["Ms A"]["decided"] == 4 and t["Mr B"]["decided"] == 3 and t["Mr B"]["dismissed_share"] == round(1 / 3, 3)
    assert rep["totals"]["missed"] == 2
    assert learning.report_csv(rep).count("\n") == 3                                    # header + head_turn + other
    assert "Student" not in json.dumps(rep)


# ------------------------------------------------------------- 6. suggestions
def test_no_suggestion_from_a_handful_of_alerts(env):
    cfg, store = env
    decide(store, cfg, learning.MIN_DECIDED - 1, "dismiss", "clock")
    assert learning.propose(store, cfg) == []


def test_many_normal_behaviour_dismissals_propose_a_less_sensitive_setting(env):
    cfg, store = env
    decide(store, cfg, 15, "dismiss", "clock")
    decide(store, cfg, 5, "dismiss", "stretching")
    decide(store, cfg, 5, "confirm")
    made = learning.propose(store, cfg)
    assert len(made) == 1
    s = made[0]
    assert (s["setting"], s["current"], s["suggested"], s["direction"], s["status"]) == ("yaw_threshold_deg", 35.0, 40.0, "less_sensitive", "pending")
    assert s["evidence"]["decided"] == 25 and s["evidence"]["normal_behaviour"] == 20
    assert cfg.yaw_threshold_deg == 35.0                                                # nothing changed by itself
    assert learning.propose(store, cfg) == []                                           # no duplicate while one is pending


def test_bad_detection_and_unexplained_dismissals_do_not_push_thresholds(env):
    cfg, store = env
    decide(store, cfg, 15, "dismiss", "bad_detection")
    decide(store, cfg, 10, "dismiss", None)
    assert learning.propose(store, cfg) == []


def test_teachers_who_disagree_get_no_suggestion(env):
    cfg, store = env
    decide(store, cfg, 12, "dismiss", "clock", "Strict Ms A")                           # dismisses everything
    decide(store, cfg, 12, "confirm", "", "Mr B")                                       # confirms everything
    assert learning.build_report(store, room=cfg.room_name)["rules"][0]["teacher_spread"] == 1.0
    assert learning.propose(store, cfg) == []


def test_missed_reports_with_few_dismissals_propose_more_sensitive(env):
    cfg, store = env
    decide(store, cfg, 22, "confirm", "", "Ms A", kind="phone")
    for _ in range(3):
        store.add_missed(cfg.room_name, "phone", "Ms A")
    made = learning.propose(store, cfg)
    assert [(m["setting"], m["direction"], m["suggested"]) for m in made] == [("phone_confidence", "more_sensitive", 0.35)]


def test_suggestion_stays_inside_safe_limits(env):
    cfg, store = env
    cfg.yaw_threshold_deg = 60.0                                                        # already at the upper limit
    decide(store, cfg, 25, "dismiss", "clock")
    assert learning.propose(store, cfg) == []


def test_approving_writes_an_override_applied_only_at_next_start(env, tmp_path, monkeypatch):
    cfg, store = env
    decide(store, cfg, 25, "dismiss", "thinking")
    s, = learning.propose(store, cfg)
    out = learning.approve(store, cfg, s["id"], "Admin")
    assert out["status"] == "approved" and out["decided_by"] == "Admin"
    assert cfg.yaw_threshold_deg == 35.0                                                # the running config is untouched
    assert learning.read_overrides(cfg.data_path) == {"yaw_threshold_deg": 40.0}
    monkeypatch.delenv("EXAMGUARD_YAW_THRESHOLD_DEG", raising=False)
    assert Config.from_env(data_dir=str(tmp_path)).yaw_threshold_deg == 40.0            # ...but the next start uses it
    monkeypatch.setenv("EXAMGUARD_YAW_THRESHOLD_DEG", "33")
    assert Config.from_env(data_dir=str(tmp_path)).yaw_threshold_deg == 33.0            # .env / environment still wins
    with pytest.raises(ReviewError):
        learning.approve(store, cfg, s["id"], "Admin")                                  # already decided
    assert learning.propose(store, cfg) == []                                           # old data is not reused after a decision


def test_rejecting_changes_nothing(env):
    cfg, store = env
    decide(store, cfg, 25, "dismiss", "clock")
    s, = learning.propose(store, cfg)
    assert learning.reject(store, cfg, s["id"], "Admin")["status"] == "rejected"
    assert learning.read_overrides(cfg.data_path) == {}


def test_only_listed_admins_can_approve_when_set(env):
    cfg, store = env
    cfg.admin_names = "Head Teacher, Ms Admin"
    decide(store, cfg, 25, "dismiss", "clock")
    s, = learning.propose(store, cfg)
    with pytest.raises(ReviewError):
        learning.approve(store, cfg, s["id"], "Random Teacher")
    assert store.get_suggestion(s["id"])["status"] == "pending"
    assert learning.approve(store, cfg, s["id"], "ms admin")["status"] == "approved"


def test_override_file_is_validated(tmp_path):
    (tmp_path / learning.OVERRIDES_FILE).write_text(json.dumps(
        {"yaw_threshold_deg": 41, "glance_count": 6.0, "password": "x", "gaze_count": "9", "phone_confidence": 5}))
    assert learning.read_overrides(tmp_path) == {"yaw_threshold_deg": 41.0, "glance_count": 6}
    (tmp_path / learning.OVERRIDES_FILE).write_text("{broken")
    assert learning.read_overrides(tmp_path) == {}


def test_old_database_without_learning_columns_is_upgraded(tmp_path):
    db = sqlite3.connect(tmp_path / "old.db")
    db.executescript("""CREATE TABLE incidents (id INTEGER PRIMARY KEY AUTOINCREMENT, created_ts REAL NOT NULL,
        room TEXT NOT NULL, track_id INTEGER NOT NULL, kind TEXT NOT NULL, severity TEXT NOT NULL, score REAL NOT NULL,
        details TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', reviewer TEXT, reviewed_ts REAL, note TEXT,
        clip_name TEXT, thumb_name TEXT, purged_ts REAL);
        INSERT INTO incidents (created_ts, room, track_id, kind, severity, score, details) VALUES (1, 'R', 1, 'phone', 'high', .9, 'd');""")
    db.commit(); db.close()
    store = Store(tmp_path / "old.db")
    old = store.list()[0]
    assert old["exam_id"] is None and old["dismiss_reason"] is None and old["ignored"] is False
    assert store.review(old["id"], "dismiss", "Ms A", reason="clock")["dismiss_reason"] == "clock"


# ---------------------------------------------------------------------- 7. web
def test_web_flow_reasons_ignore_exam_missed_report_and_suggestions(tmp_path):
    cfg = Config(data_dir=str(tmp_path), password="pw")
    svc = build_services(cfg, demo=True, demo_auto=False)
    app = create_app(svc); c = app.test_client()
    assert c.get("/exam").status_code == 302 and c.post("/api/missed", json={}).status_code == 401
    assert c.post("/login", data={"name": "Ms A", "password": "pw"}).status_code == 302
    assert c.get("/exam").status_code == 200 and b"Finish exam" in c.get("/exam").data
    assert b"Ignore for now" in c.get("/").data

    reasons = c.get("/api/reasons").json
    assert {r["key"] for r in reasons["reasons"]} >= {"clock", "bad_detection", "other"}

    a = svc.store.add_incident(ev(), cfg.room_name, None, None)
    assert c.post(f"/api/incidents/{a['id']}/review", json={"decision": "dismiss", "reason": "nope"}).status_code == 400
    assert c.post(f"/api/incidents/{a['id']}/review", json={"decision": "confirm", "reason": "clock"}).status_code == 400
    assert svc.store.get(a["id"])["status"] == "pending"
    r = c.post(f"/api/incidents/{a['id']}/review", json={"decision": "dismiss", "reason": "clock"})
    assert r.status_code == 200 and r.json["dismiss_reason"] == "clock"
    assert c.post(f"/api/incidents/{a['id']}/reason", json={"reason": "thinking"}).json["dismiss_reason"] == "thinking"

    b = svc.store.add_incident(ev(), cfg.room_name, None, None)
    assert c.post(f"/api/incidents/{b['id']}/ignore", json={"on": True}).json["ignored"] is True
    assert c.post(f"/api/incidents/{a['id']}/ignore", json={}).status_code == 409
    assert c.post("/api/incidents/999/ignore", json={}).status_code == 404

    assert c.post("/api/missed", json={"kind": "phone", "minutes_ago": 3, "note": "back row"}).status_code == 200
    assert c.post("/api/missed", json={"kind": "zzz"}).status_code == 400
    cur = c.get("/api/exam").json["current"]
    assert cur["counts"]["missed"] == 1 and cur["counts"]["waiting"] == 1
    done = c.post("/api/exam/finish", json={}).json
    assert done["counts"]["dismissed"] == 1 and not done["exam"]["open"]
    assert c.post("/api/exam/finish", json={}).status_code == 409
    assert c.get(f"/api/exams/{done['exam']['id']}").status_code == 200 and c.get("/api/exams/999").status_code == 404

    rep = c.get("/api/learning/report").json
    assert rep["rules"][0]["dismissed"] == 1 and rep["totals"]["missed"] == 1
    assert b"head_turn" in c.get("/api/learning/report.csv").data

    for _ in range(25):
        i = svc.store.add_incident(ev(), cfg.room_name, None, None)
        c.post(f"/api/incidents/{i['id']}/review", json={"decision": "dismiss", "reason": "stretching"})
    assert c.post("/api/learning/suggestions/refresh").json["new"] == 1
    sug = c.get("/api/learning/suggestions").json
    assert sug["can_approve"] and sug["suggestions"][0]["setting"] == "yaw_threshold_deg"
    sid = sug["suggestions"][0]["id"]
    ok = c.post(f"/api/learning/suggestions/{sid}/approve")
    assert ok.status_code == 200 and ok.json["restart_needed"] and svc.cfg.yaw_threshold_deg == 35.0
    assert c.post(f"/api/learning/suggestions/{sid}/approve").status_code == 409
    assert c.post(f"/api/learning/suggestions/{sid}/explode").status_code == 404
    assert c.get("/api/learning/suggestions").json["overrides"] == {"yaw_threshold_deg": 40.0}
    actions = {x["action"] for x in c.get("/api/audit?limit=1000").json["audit"]}
    assert {"ignored", "missed_reported", "exam_finished", "suggestion_approved", "reason_corrected"} <= actions
    svc.stop()
