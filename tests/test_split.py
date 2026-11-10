"""Edge box <-> server split over MQTT, end to end, against the in-process test broker (see tests/minibroker.py).
Never run against a real Mosquitto broker or real hardware."""
import base64
import json
import time

import pytest
from cryptography.fernet import Fernet

from examguard import splitmode
from examguard.config import Config
from examguard.models import Event
from examguard.mqttlite import MqttClient
from examguard.seatmap import Seat
from examguard.service import build_services
from examguard.splitmode import (EdgeLink, WireError, Wire, build_incident_message, event_from_dict, event_to_dict,
                                 room_slug, seatmap_digest)
from examguard.web.app import create_app
from tests.minibroker import MiniBroker, wait_for

KEY = Fernet.generate_key().decode()


def cfgs(tmp_path, port, **kw):
    base = dict(password="pw", room_name="Room 1", mqtt_host="127.0.0.1", mqtt_port=port, mqtt_key=KEY,
                heartbeat_seconds=0.3, live_seconds=0.3, node_name="pi-test")
    base.update(kw)
    return (Config(data_dir=str(tmp_path / "server"), role="server", **base),
            Config(data_dir=str(tmp_path / "edge"), role="edge", **base))


@pytest.fixture
def pair(tmp_path):
    broker = MiniBroker()
    scfg, ecfg = cfgs(tmp_path, broker.port)
    server = build_services(scfg)
    server.start()
    edge = build_services(ecfg, demo=True, demo_auto=False)
    edge.start()
    app = create_app(server)
    client = app.test_client()
    client.post("/login", data={"name": "Ms A", "password": "pw"})
    yield broker, server, edge, client
    edge.stop()
    server.stop()
    broker.stop()


def raise_alert(edge, kind="head_turn", student=2):
    time.sleep(4.5)                                    # each student's usual posture forms first
    edge.worlds[0].trigger(kind, student=student)


# -------------------------------------------------------------------- the wire
def test_a_key_is_required_and_validated():
    with pytest.raises(ValueError, match="make_key"):
        Wire("")
    with pytest.raises(ValueError, match="not a valid key"):
        Wire("not-a-key")


def test_sealed_messages_are_authenticated():
    w = Wire(KEY)
    token = w.seal(b"clip bytes")
    assert b"clip bytes" not in token and w.open(token) == b"clip bytes"
    with pytest.raises(WireError):
        w.open(token[:-3] + b"xyz")                                      # tampered
    with pytest.raises(WireError, match="MQTT_KEY"):
        Wire(Fernet.generate_key().decode()).open(token)                 # other key
    with pytest.raises(WireError):
        w.open_json(w.seal(b"{not json"))


def test_event_round_trip_and_strict_validation():
    ev = Event(ts=time.time(), track_id=3, kind="phone", severity="high", score=0.9, details="d", bbox=(1.0, 2.0, 3.0, 4.0),
               student_name="Layla", student_ref="r1", match_score=0.8, seat="B2", seat_student="Layla", id_source="face",
               camera="Front camera")
    back = event_from_dict(json.loads(json.dumps(event_to_dict(ev))))
    assert back == ev
    good = event_to_dict(ev)
    for patch in ({"kind": "teleport"}, {"severity": "extreme"}, {"score": 7}, {"ts": -1}, {"bbox": [1, 2]},
                  {"student_name": 5}, {"track_id": "x"}):
        with pytest.raises(WireError):
            event_from_dict({**good, **patch})
    with pytest.raises(WireError):
        event_from_dict("nope")


def test_room_slug_and_topics_are_safe():
    assert room_slug("Exam Room 1") == "exam-room-1" and room_slug("../#+") == "room" and room_slug("غرفة") == "room"


def test_misconfiguration_is_refused_with_plain_messages(tmp_path):
    with pytest.raises(ValueError, match="ROLE"):
        build_services(Config(data_dir=str(tmp_path / "a"), role="banana"))
    with pytest.raises(ValueError, match="MQTT_KEY"):
        build_services(Config(data_dir=str(tmp_path / "b"), role="edge", mqtt_host="x"), demo=True)
    with pytest.raises(ValueError, match="demo"):
        build_services(Config(data_dir=str(tmp_path / "c"), role="server", mqtt_key=KEY, mqtt_host="x"), demo=True)
    with pytest.raises(ValueError, match="MQTT_HOST"):
        build_services(Config(data_dir=str(tmp_path / "d"), role="server", mqtt_key=KEY))


# ------------------------------------------------------------------ end to end
def test_an_alert_travels_from_the_edge_to_the_servers_review_queue(pair):
    broker, server, edge, client = pair
    seen = server.hub.subscribe()
    raise_alert(edge)
    inc = wait_for(lambda: (server.store.list("pending") or [None])[0], 20)
    assert inc, "the alert never reached the server"
    assert inc["kind"] == "head_turn" and inc["status"] == "pending" and inc["has_clip"]
    assert inc["seat"] and inc["student"]                                 # seat + roster match were done on the edge
    names = server.store.media_names(inc["id"])
    assert len(server.vault.get(names["clip"])) > 1000 and server.vault.get(names["thumb"])[:2] == b"\xff\xd8"
    assert client.get(f"/media/{inc['id']}/clip").status_code == 200   # a teacher can watch it
    assert any(a["action"] == "edge_received" and "pi-test" in a["detail"] for a in server.store.list_audit())
    assert seen.get(timeout=2)[0] == "incident"                           # dashboards are told, like a local alert
    # nothing is left on the edge box once delivered
    assert wait_for(lambda: not edge.link.pending(), 10)
    assert list(edge.vault.root.glob("*.enc")) == []
    # a human still decides
    assert client.post(f"/api/incidents/{inc['id']}/review", json={"decision": "dismiss", "reason": "clock"}).status_code == 200


def test_status_live_picture_snapshot_and_health_come_from_the_edge(pair):
    broker, server, edge, client = pair
    st = wait_for(lambda: (lambda d: d if d["cameras"][0]["state"] == "running" else None)(client.get("/api/status").get_json()), 15)
    assert st and st["role"] == "server" and st["cameras"][0]["students"] >= 1 and st["edge"]["connected"]
    jpeg, fid = server.monitor.wait_frame(-1, 10)
    assert jpeg and jpeg[:2] == b"\xff\xd8" and fid >= 1
    assert wait_for(lambda: server.monitor.snapshot_jpeg(), 10) and server.monitor.snapshot_faces()
    assert client.get("/api/seatmap/snapshot").status_code == 200
    health = client.get("/api/health").get_json()
    by = {i["title"]: i for i in health["items"]}
    assert by["Link to the broker"]["level"] == "ok" and by["Edge box"]["level"] == "ok"
    assert by["Live view"]["level"] == "info" and by["Clocks"]["level"] == "ok" if "Clocks" in by else True


def test_the_edge_box_going_quiet_shows_as_offline(pair, monkeypatch):
    broker, server, edge, client = pair
    assert wait_for(lambda: server.monitor.status()["state"] == "running", 15)
    monkeypatch.setattr(splitmode, "OFFLINE_AFTER", 1.0)
    edge.link.stop()                                                      # heartbeats stop
    assert wait_for(lambda: server.monitor.status()["state"] == "offline", 10)
    by = {i["title"]: i for i in client.get("/api/health").get_json()["items"]}
    assert by["Edge box"]["level"] == "fail" and "no report" in by["Edge box"]["detail"], (by["Edge box"], server.link.info())
    assert not any(t.endswith("video") for t in by)                       # no misleading "camera cable" advice


def test_clock_differences_and_a_second_edge_box_are_flagged(pair):
    broker, server, edge, client = pair
    assert wait_for(lambda: server.link.feed.status, 10)
    status = dict(server.link.feed.status, now=time.time() + 90, node="pi-other")
    server.link.feed.update_status(status)
    by = {i["title"]: i for i in client.get("/api/health").get_json()["items"]}
    assert by["Clocks"]["level"] == "warn" and by["Several edge boxes"]["level"] == "warn"


# ------------------------------------------------------------------ reliability
def test_alerts_wait_in_the_outbox_while_the_broker_is_down_and_arrive_once(tmp_path):
    broker = MiniBroker()
    scfg, ecfg = cfgs(tmp_path, broker.port)
    server = build_services(scfg); server.start()
    edge = build_services(ecfg, demo=True, demo_auto=False); edge.start()
    try:
        assert wait_for(lambda: server.link.client.connected and edge.link.client.connected, 10)
        time.sleep(4.5)
        broker.stop()                                                     # the network / broker dies
        assert wait_for(lambda: not edge.link.client.connected, 5)
        edge.worlds[0].trigger("head_turn", student=2)
        assert wait_for(lambda: len(edge.link.pending()) == 1, 20), "the alert must be kept on the edge box"
        assert server.store.list("pending") == []
        broker.start()                                                    # same port, sessions kept like persistence
        assert wait_for(lambda: len(server.store.list("pending")) == 1, 75)      # reconnect back-off grows to 15 s
        time.sleep(1.5)
        assert len(server.store.list("pending")) == 1 and not edge.link.pending()
    finally:
        edge.stop(); server.stop(); broker.stop()


def test_a_lost_acknowledgement_makes_the_edge_resend_but_never_duplicates(pair):
    broker, server, edge, client = pair
    edge.link.ack_timeout = 1.0
    broker.drop_next_puback_for.add(edge.link.topics.incident)            # the edge will not hear the first ack
    raise_alert(edge)
    assert wait_for(lambda: server.link.duplicates >= 1, 25), "the edge should have sent the alert a second time"
    assert wait_for(lambda: not edge.link.pending(), 10)
    assert len(server.store.list("pending")) == 1


def test_alerts_sent_while_the_server_is_away_are_delivered_when_it_returns(pair):
    broker, server, edge, client = pair
    time.sleep(4.5)
    server.link.client.stop()                                             # the teacher's computer sleeps
    time.sleep(0.5)
    edge.worlds[0].trigger("head_turn", student=2)
    assert wait_for(lambda: not edge.link.pending(), 20)                  # the BROKER holds it (persistent session)
    assert server.store.list("pending") == []
    server.link.client.start()
    assert wait_for(lambda: len(server.store.list("pending")) == 1, 20)


def test_a_message_with_the_wrong_key_is_rejected_and_does_not_block_real_alerts(pair):
    broker, server, edge, client = pair
    intruder = MqttClient("127.0.0.1", broker.port, "intruder")
    intruder.start(); assert intruder.wait_connected(5)
    ev = Event(ts=time.time(), track_id=1, kind="phone", severity="high", score=0.9, details="fake")
    _, forged = build_incident_message(Wire(Fernet.generate_key().decode()), "evil", "Room 1", ev, None, None)
    assert intruder.publish(server.link.topics.incident, forged, qos=1)
    assert wait_for(lambda: server.link.rejected == 1, 10)
    assert server.store.list("pending") == []
    assert any(a["action"] == "edge_message_rejected" for a in server.store.list_audit())
    raise_alert(edge)
    assert wait_for(lambda: len(server.store.list("pending")) == 1, 20)
    intruder.stop()


def test_malformed_but_correctly_sealed_alerts_are_rejected(pair):
    broker, server, edge, client = pair
    w = Wire(KEY)
    ok_ev = event_to_dict(Event(ts=time.time(), track_id=1, kind="phone", severity="high", score=0.9, details="d"))
    bad = [{"v": 1, "id": "../../etc/passwd", "event": ok_ev},
           {"v": 1, "id": "a" * 32, "event": {**ok_ev, "kind": "nope"}},
           {"v": 2, "id": "b" * 32, "event": ok_ev},
           {"v": 1, "id": "c" * 32, "event": ok_ev, "clip": "###not base64###"}]
    for b in bad:
        with pytest.raises(WireError):
            server.link.ingest_incident(w.seal_json(b))
    assert server.store.list() == [] and list(server.vault.root.glob("clip-r*")) == []
    good = w.seal_json({"v": 1, "id": "d" * 32, "event": ok_ev, "clip": base64.b64encode(b"x" * 50).decode(), "node": "n"})
    assert server.link.ingest_incident(good)["kind"] == "phone"
    assert server.link.ingest_incident(good) is None                      # same id again: ignored
    assert len(server.store.list()) == 1


def test_the_outbox_is_capped_and_drops_the_oldest_alert_loudly(tmp_path):
    broker = MiniBroker(); broker.stop()                                  # nobody to send to
    _, ecfg = cfgs(tmp_path, broker.port, outbox_max_mb=1)
    edge = build_services(ecfg, demo=True, demo_auto=False)
    ev = Event(ts=time.time(), track_id=1, kind="phone", severity="high", score=0.9, details="d")
    big = edge.vault.put("clip-big.mp4", b"\0" * 700_000)
    ids = [edge.link.submit_incident(ev, edge.vault.put(f"clip-{i}.mp4", b"\x01" * 700_000), None) for i in range(3)]
    assert edge.link.dropped >= 1 and len(edge.link.pending()) < 3
    assert all(i in n.name for i, n in zip(ids[-len(edge.link.pending()):], edge.link.pending()))   # newest are kept
    assert edge.link.status_message()["dropped"] == edge.link.dropped
    edge.store.close()


# -------------------------------------------------------------------- seat map
def test_the_servers_seat_map_is_sent_to_the_edge_box_and_heals_after_a_broker_restart(pair):
    broker, server, edge, client = pair
    assert wait_for(lambda: server.link.client.connected and edge.link.client.connected, 10)
    mine = [{"label": "Z1", "x": 0.2, "y": 0.3, "camera": 0}, {"label": "Z2", "x": 0.6, "y": 0.3, "camera": 0}]
    r = client.put("/api/seatmap", json={"seats": mine})
    assert r.status_code == 200 and "edge box" in r.get_json()["note"]
    assert wait_for(lambda: [s.label for s in edge.seatmap.seats()] == ["Z1", "Z2"], 10)
    # the broker forgets its retained message and the edge box is changed behind our back
    broker.retained.clear()
    edge.seatmap.replace([Seat("Q9", 0.5, 0.5, 0, "")])
    assert wait_for(lambda: [s.label for s in edge.seatmap.seats()] == ["Z1", "Z2"], 45), "the server should push its map again"
    assert seatmap_digest(edge.seatmap.seats()) == seatmap_digest(server.seatmap.seats())


def test_an_invalid_seat_map_from_the_broker_is_ignored_by_the_edge(pair):
    broker, server, edge, client = pair
    before = seatmap_digest(edge.seatmap.seats())
    forged = Wire(KEY).seal_json({"seats": [{"label": "<script>", "x": 5, "y": 5}]})
    server.link.client.publish(server.link.topics.seatmap, forged, qos=1, retain=False)
    time.sleep(1.0)
    assert seatmap_digest(edge.seatmap.seats()) == before


def test_standalone_mode_is_unchanged(tmp_path):
    svc = build_services(Config(data_dir=str(tmp_path), password="pw"), demo=True, demo_auto=False)
    assert svc.role == "standalone" and svc.link is None
    svc.store.close()
