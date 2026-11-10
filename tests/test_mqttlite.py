"""The dependency-free MQTT client, against the in-process test broker (see tests/minibroker.py)."""
import os
import threading
import time

import pytest

from examguard.mqttlite import MqttClient, encode_length
from tests.minibroker import MiniBroker, topic_matches, wait_for


@pytest.fixture
def broker():
    b = MiniBroker()
    yield b
    b.stop()


def client(b, cid, handler=None, **kw):
    c = MqttClient("127.0.0.1", b.port, cid, on_message=handler, **kw)
    c.start()
    assert c.wait_connected(5), c.last_error
    return c


def test_length_encoding_matches_the_spec():
    assert encode_length(0) == b"\x00" and encode_length(127) == b"\x7f"
    assert encode_length(128) == b"\x80\x01" and encode_length(16383) == b"\xff\x7f"
    assert encode_length(16384) == b"\x80\x80\x01" and encode_length(2097152) == b"\x80\x80\x80\x01"


def test_wildcards():
    assert topic_matches("a/+/c", "a/b/c") and topic_matches("a/#", "a/b/c") and not topic_matches("a/+/c", "a/b/d")
    assert not topic_matches("a/+", "a/b/c")


def test_publish_qos1_waits_for_the_ack_and_subscriber_receives(broker):
    got = []
    sub = client(broker, "sub", lambda t, p, r: got.append((t, p, r)))
    sub.subscribe("ex/+/incident", 1)
    time.sleep(0.3)
    pub = client(broker, "pub")
    assert pub.publish("ex/room1/incident", b"hello", qos=1)
    assert wait_for(lambda: got) == [("ex/room1/incident", b"hello", False)]
    assert pub.publish("ex/room1/other", b"x", qos=0)
    time.sleep(0.2)
    assert len(got) == 1                                          # not subscribed to that topic
    sub.stop(); pub.stop()


def test_large_payload_and_binary_bytes_survive(broker):
    got = []
    sub = client(broker, "sub", lambda t, p, r: got.append(p))
    sub.subscribe("big", 1)
    time.sleep(0.3)
    data = os.urandom(3 * 1024 * 1024)
    pub = client(broker, "pub")
    assert pub.publish("big", data, qos=1, timeout=20)
    assert wait_for(lambda: got, 20) == [data]
    sub.stop(); pub.stop()


def test_retained_message_reaches_a_late_subscriber(broker):
    pub = client(broker, "pub")
    assert pub.publish("cfg/seatmap", b"v1", qos=1, retain=True)
    got = []
    sub = client(broker, "late", lambda t, p, r: got.append((p, r)))
    sub.subscribe("cfg/#", 1)
    assert wait_for(lambda: got) == [(b"v1", True)]
    sub.stop(); pub.stop()


def test_wrong_password_is_refused_with_a_clear_reason():
    b = MiniBroker(users={"edge": "secret"})
    try:
        c = MqttClient("127.0.0.1", b.port, "x", username="edge", password="nope")
        c.start()
        assert not c.wait_connected(1.5)
        assert "user name or password" in c.last_error
        c.stop()
        ok = MqttClient("127.0.0.1", b.port, "y", username="edge", password="secret")
        ok.start()
        assert ok.wait_connected(5)
        ok.stop()
    finally:
        b.stop()


def test_publish_returns_false_when_the_broker_never_acknowledges(broker):
    pub = client(broker, "pub")
    broker.drop_next_puback_for.add("t")
    assert pub.publish("t", b"x", qos=1, timeout=0.5) is False
    assert pub.publish("t", b"x", qos=1, timeout=3) is True
    pub.stop()


def test_publish_when_not_connected_is_false():
    c = MqttClient("127.0.0.1", 1, "nobody")
    assert c.publish("t", b"x", qos=1) is False


def test_reconnects_and_resubscribes_after_the_connection_is_cut(broker):
    got = []
    sub = client(broker, "sub", lambda t, p, r: got.append(p))
    sub.subscribe("t", 1)
    time.sleep(0.3)
    broker.cut_connections()
    assert wait_for(lambda: not sub.connected, 3)
    assert sub.wait_connected(10)
    pub = client(broker, "pub")
    assert pub.publish("t", b"after", qos=1)
    assert wait_for(lambda: got) == [b"after"]
    sub.stop(); pub.stop()


def test_a_handler_that_raises_gets_the_message_again_and_only_then_acks(broker):
    calls = []
    def handler(t, p, r):
        calls.append(p)
        if len(calls) == 1:
            raise RuntimeError("disk full")
    sub = client(broker, "sub", handler, clean_session=False)
    sub.subscribe("t", 1)
    time.sleep(0.3)
    pub = client(broker, "pub")
    assert pub.publish("t", b"m1", qos=1)
    assert wait_for(lambda: len(calls) == 1)
    time.sleep(0.2)
    broker.cut_connections()                                      # the broker re-sends unacknowledged QoS 1 on reconnect
    assert wait_for(lambda: len(calls) == 2, 15)
    assert calls == [b"m1", b"m1"]
    sub.stop(); pub.stop()


def test_messages_sent_while_a_persistent_subscriber_is_away_are_kept(broker):
    got = []
    sub = client(broker, "srv", lambda t, p, r: got.append(p), clean_session=False)
    sub.subscribe("t", 1)
    time.sleep(0.3)
    sub.stop()
    pub = client(broker, "pub")
    assert pub.publish("t", b"while-away", qos=1)
    sub2 = MqttClient("127.0.0.1", broker.port, "srv", on_message=lambda t, p, r: got.append(p), clean_session=False)
    sub2.subscribe("t", 1)
    sub2.start()
    assert wait_for(lambda: got) == [b"while-away"]
    sub2.stop(); pub.stop()
