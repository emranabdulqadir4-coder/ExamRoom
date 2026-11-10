"""A tiny in-process MQTT 3.1.1 broker for the tests (NOT part of ExamGuard, and not a real Mosquitto).

Supports: optional username/password, QoS 0/1, '+' and '#' wildcards, retained messages, persistent sessions
(QoS 1 messages queue while a client is away and are re-sent until acknowledged), and test hooks to cut every
connection or to stop and restart the broker.
"""
import socket
import struct
import threading
import time

from examguard.mqttlite import (CONNACK, CONNECT, DISCONNECT, PINGREQ, PINGRESP, PUBACK, PUBLISH, SUBACK, SUBSCRIBE,
                                MqttError, packet, parse_publish, read_packet, utf8)


def topic_matches(filt: str, topic: str) -> bool:
    f, t = filt.split("/"), topic.split("/")
    for i, part in enumerate(f):
        if part == "#":
            return True
        if i >= len(t) or (part != "+" and part != t[i]):
            return False
    return len(f) == len(t)


class _Session:
    def __init__(self, client_id: str, clean: bool):
        self.client_id, self.clean = client_id, clean
        self.subs = {}                  # filter -> qos
        self.queue = []                 # (topic, payload, qos, retain) waiting for delivery
        self.unacked = {}               # pid -> (topic, payload)
        self.conn = None                # current _Conn or None
        self.next_pid = 0


class _Conn:
    def __init__(self, sock, session):
        self.sock, self.session = sock, session
        self.lock = threading.Lock()

    def send(self, data: bytes) -> None:
        with self.lock:
            self.sock.sendall(data)


class MiniBroker:
    def __init__(self, port: int = 0, users=None):
        self.users = users or {}            # {} = anonymous allowed
        self._lock = threading.RLock()
        self.sessions = {}
        self.retained = {}
        self.published = []                 # (topic, payload) in arrival order, for the tests
        self.drop_next_puback_for = set()   # topics whose PUBACK to the publisher is withheld once
        self._srv = None
        self.port = port
        self._threads = []
        self.start()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", self.port))
        self._srv.listen(16)
        self.port = self._srv.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def stop(self) -> None:
        """Close the listener and every connection (sessions and retained messages survive, like persistence)."""
        srv, self._srv = self._srv, None
        if srv:
            try:
                srv.shutdown(socket.SHUT_RDWR)          # wakes the thread blocked in accept()
            except OSError:
                pass
            try:
                srv.close()
            except OSError:
                pass
        self.cut_connections()

    def cut_connections(self) -> None:
        with self._lock:
            conns = [s.conn for s in self.sessions.values() if s.conn]
        for c in conns:
            try:
                c.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                c.sock.close()
            except OSError:
                pass

    def _accept(self) -> None:
        srv = self._srv
        while srv is not None and self._srv is srv:
            try:
                sock, _ = srv.accept()
            except OSError:
                return
            if self._srv is not srv:                    # stopped while we were waiting: a stopped broker takes no one
                sock.close()
                return
            threading.Thread(target=self._serve, args=(sock,), daemon=True).start()

    # -------------------------------------------------------------- serving
    def _serve(self, sock) -> None:
        conn = None
        try:
            ptype, _, body = read_packet(sock)
            if ptype != CONNECT:
                return
            (nlen,) = struct.unpack("!H", body[:2])
            pos = 2 + nlen + 1                          # protocol name + level
            flags = body[pos]
            pos += 1 + 2                                # flags + keepalive
            def take(p):
                (n,) = struct.unpack("!H", body[p:p + 2])
                return body[p + 2:p + 2 + n].decode(), p + 2 + n
            client_id, pos = take(pos)
            user = pw = ""
            if flags & 0x80:
                user, pos = take(pos)
            if flags & 0x40:
                pw, pos = take(pos)
            if self.users and self.users.get(user) != pw:
                sock.sendall(packet(CONNACK, 0, bytes([0, 4])))
                return
            clean = bool(flags & 0x02)
            with self._lock:
                sess = self.sessions.get(client_id)
                if sess is None or clean:
                    sess = self.sessions[client_id] = _Session(client_id, clean)
                if sess.conn:
                    try:
                        sess.conn.sock.close()
                    except OSError:
                        pass
                conn = sess.conn = _Conn(sock, sess)
            conn.send(packet(CONNACK, 0, bytes([0, 0])))
            self._redeliver(sess)
            while True:
                ptype, fl, body = read_packet(sock)
                if ptype == PUBLISH:
                    self._on_publish(conn, fl, body)
                elif ptype == SUBSCRIBE:
                    (pid,) = struct.unpack("!H", body[:2])
                    pos, granted, new = 2, b"", []
                    while pos < len(body):
                        filt, pos = take_from(body, pos)
                        qos = min(1, body[pos]); pos += 1
                        sess.subs[filt] = qos
                        granted += bytes([qos]); new.append((filt, qos))
                    conn.send(packet(SUBACK, 0, struct.pack("!H", pid) + granted))
                    for filt, qos in new:
                        with self._lock:
                            hits = [(t, v) for t, v in self.retained.items() if topic_matches(filt, t)]
                        for t, v in hits:
                            self._deliver(sess, t, v, qos, retain=True)
                elif ptype == PUBACK:
                    (pid,) = struct.unpack("!H", body[:2])
                    sess.unacked.pop(pid, None)
                elif ptype == PINGREQ:
                    conn.send(packet(PINGRESP, 0, b""))
                elif ptype == DISCONNECT:
                    return
        except (OSError, MqttError, ConnectionError, struct.error):
            pass
        finally:
            with self._lock:
                if conn and conn.session.conn is conn:
                    conn.session.conn = None
                    if conn.session.clean:
                        self.sessions.pop(conn.session.client_id, None)
            try:
                sock.close()
            except OSError:
                pass

    def _on_publish(self, conn, flags, body) -> None:
        topic, qos, pid, payload, retain, _ = parse_publish(flags, body)
        with self._lock:
            self.published.append((topic, payload))
            if retain:
                if payload:
                    self.retained[topic] = payload
                else:
                    self.retained.pop(topic, None)
            targets = list(self.sessions.values())
        for sess in targets:
            for filt, sqos in list(sess.subs.items()):
                if topic_matches(filt, topic):
                    self._deliver(sess, topic, payload, min(qos, sqos), retain=False)
                    break
        if qos == 1 and pid is not None:
            if topic in self.drop_next_puback_for:
                self.drop_next_puback_for.discard(topic)
                return
            conn.send(packet(PUBACK, 0, struct.pack("!H", pid)))

    def _deliver(self, sess, topic, payload, qos, retain) -> None:
        with self._lock:
            conn = sess.conn
            if qos == 0:
                if conn is None:
                    return
                flags = 1 if retain else 0
                data = packet(PUBLISH, flags, utf8(topic) + payload)
            else:
                sess.next_pid = sess.next_pid % 65535 + 1
                pid = sess.next_pid
                sess.unacked[pid] = (topic, payload)
                data = packet(PUBLISH, (1 << 1) | (1 if retain else 0), utf8(topic) + struct.pack("!H", pid) + payload)
        if conn is not None:
            try:
                conn.send(data)
            except OSError:
                pass

    def _redeliver(self, sess) -> None:
        with self._lock:
            pending = list(sess.unacked.items())
            sess.unacked.clear()
        for pid, (topic, payload) in pending:
            self._deliver(sess, topic, payload, 1, retain=False)


def take_from(body: bytes, pos: int):
    (n,) = struct.unpack("!H", body[pos:pos + 2])
    return body[pos + 2:pos + 2 + n].decode(), pos + 2 + n


def wait_for(fn, timeout=8.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(step)
    return fn()
