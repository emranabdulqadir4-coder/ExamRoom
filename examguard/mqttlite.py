"""A small MQTT 3.1.1 client with no extra dependencies (Phase 2: edge box <-> server link).

It implements only what ExamGuard needs: username/password, optional TLS, QoS 0 and QoS 1 publish (QoS 1 waits for
the broker's acknowledgement), subscribe, retained messages, keep-alive pings and automatic reconnection.

HONEST STATUS: tested only against the small in-process test broker in ``tests/minibroker.py`` (which was
written from the same specification, so a misreading would be shared). It has NOT been run against a real
Mosquitto broker. If you prefer a mature library, this class is the only place that would change.
"""
from __future__ import annotations

import logging
import socket
import ssl
import struct
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger("examguard.mqtt")

MAX_PACKET = 64 * 1024 * 1024          # refuse anything bigger (a broken or hostile peer)

CONNECT, CONNACK, PUBLISH, PUBACK = 1, 2, 3, 4
SUBSCRIBE, SUBACK, PINGREQ, PINGRESP, DISCONNECT = 8, 9, 12, 13, 14

# handler(topic, payload, retained) -> None. Returning normally acknowledges a QoS 1 message;
# raising means "not stored": no acknowledgement is sent, so the broker will deliver it again.
Handler = Callable[[str, bytes, bool], None]


class MqttError(Exception):
    pass


def encode_length(n: int) -> bytes:
    out = bytearray()
    while True:
        digit, n = n % 128, n // 128
        out.append(digit | (0x80 if n else 0))
        if not n:
            return bytes(out)


def utf8(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("!H", len(b)) + b


def read_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def read_packet(sock, first: Optional[int] = None) -> Tuple[int, int, bytes]:
    """(packet type, flags, body). ``first`` is the first byte if the caller already read it."""
    if first is None:
        first = read_exact(sock, 1)[0]
    mult, length = 1, 0
    for _ in range(4):
        b = read_exact(sock, 1)[0]
        length += (b & 0x7F) * mult
        if not b & 0x80:
            break
        mult *= 128
    else:
        raise MqttError("malformed length")
    if length > MAX_PACKET:
        raise MqttError("packet too large")
    return first >> 4, first & 0x0F, read_exact(sock, length) if length else b""


def packet(ptype: int, flags: int, body: bytes) -> bytes:
    return bytes([(ptype << 4) | flags]) + encode_length(len(body)) + body


def parse_publish(flags: int, body: bytes) -> Tuple[str, int, Optional[int], bytes, bool, bool]:
    """(topic, qos, packet id, payload, retain, dup)."""
    qos = (flags >> 1) & 3
    (tlen,) = struct.unpack("!H", body[:2])
    topic = body[2:2 + tlen].decode("utf-8")
    pos = 2 + tlen
    pid = None
    if qos:
        (pid,) = struct.unpack("!H", body[pos:pos + 2])
        pos += 2
    return topic, qos, pid, body[pos:], bool(flags & 1), bool(flags & 8)


class MqttClient:
    def __init__(self, host: str, port: int = 1883, client_id: str = "examguard", username: str = "",
                 password: str = "", tls: bool = False, cafile: str = "", keepalive: int = 30,
                 clean_session: bool = True, on_message: Optional[Handler] = None):
        self.host, self.port, self.client_id = host, int(port), client_id
        self.username, self.password = username, password
        self.tls, self.cafile = tls, cafile
        self.keepalive = max(5, int(keepalive))
        self.clean_session = clean_session
        self.on_message = on_message
        self._subs: Dict[str, int] = {}
        self._sock: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pid = 0
        self._acks: Dict[int, threading.Event] = {}
        self._last_rx = 0.0
        self.last_error = ""

    # ------------------------------------------------------------ public
    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mqtt", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock = self._sock
        if sock is not None:
            try:
                with self._send_lock:
                    sock.sendall(packet(DISCONNECT, 0, b""))
            except OSError:
                pass
            self._close(sock)
        if self._thread:
            self._thread.join(timeout=5)

    def wait_connected(self, timeout: float = 5.0) -> bool:
        return self._connected.wait(timeout)

    def subscribe(self, topic: str, qos: int = 1) -> None:
        """Remembered, and renewed automatically after every reconnect."""
        self._subs[topic] = min(1, qos)
        if self.connected:
            try:
                self._send_subscribe([(topic, min(1, qos))])
            except OSError:
                pass

    def publish(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False, timeout: float = 10.0) -> bool:
        """QoS 0: True if it was handed to the network. QoS 1: True only once the broker acknowledged it."""
        if not self.connected:
            return False
        flags = (1 if retain else 0)
        body = utf8(topic)
        ev = None
        pid = 0
        if qos >= 1:
            flags |= 1 << 1
            with self._state_lock:
                self._pid = self._pid % 65535 + 1
                pid = self._pid
                ev = self._acks[pid] = threading.Event()
            body += struct.pack("!H", pid)
        data = packet(PUBLISH, flags, body + payload)
        try:
            self._send(data)
        except OSError as exc:
            self._drop(exc)
            if ev is not None:
                self._acks.pop(pid, None)
            return False
        if ev is None:
            return True
        ok = ev.wait(timeout)
        self._acks.pop(pid, None)
        return ok

    # ----------------------------------------------------------- internals
    def _send(self, data: bytes) -> None:
        sock = self._sock
        if sock is None:
            raise OSError("not connected")
        with self._send_lock:
            sock.sendall(data)

    def _send_subscribe(self, items: List[Tuple[str, int]]) -> None:
        with self._state_lock:
            self._pid = self._pid % 65535 + 1
            pid = self._pid
        body = struct.pack("!H", pid) + b"".join(utf8(t) + bytes([q]) for t, q in items)
        self._send(packet(SUBSCRIBE, 2, body))

    def _close(self, sock) -> None:
        self._connected.clear()
        try:
            sock.close()
        except OSError:
            pass
        if self._sock is sock:
            self._sock = None

    def _drop(self, exc) -> None:
        sock = self._sock
        if sock is not None:
            self.last_error = str(exc)
            self._close(sock)

    def _connect_once(self) -> socket.socket:
        raw = socket.create_connection((self.host, self.port), timeout=10)
        sock: socket.socket = raw
        if self.tls:
            ctx = ssl.create_default_context(cafile=self.cafile or None)
            sock = ctx.wrap_socket(raw, server_hostname=self.host)
        flags = 0x02 if self.clean_session else 0
        payload = utf8(self.client_id)
        if self.username:
            flags |= 0x80
            payload += utf8(self.username)
            if self.password:
                flags |= 0x40
                payload += utf8(self.password)
        body = utf8("MQTT") + bytes([4, flags]) + struct.pack("!H", self.keepalive) + payload
        sock.sendall(packet(CONNECT, 0, body))
        ptype, _, resp = read_packet(sock)
        if ptype != CONNACK or len(resp) < 2:
            raise MqttError("the broker did not answer the connection request")
        if resp[1] != 0:
            reasons = {1: "unsupported protocol", 2: "client id refused", 3: "broker unavailable",
                       4: "bad user name or password", 5: "not authorised"}
            raise MqttError("broker refused the connection: " + reasons.get(resp[1], f"code {resp[1]}"))
        sock.settimeout(1.0)
        return sock

    def _run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            try:
                sock = self._connect_once()
            except (OSError, MqttError, ssl.SSLError) as exc:
                self.last_error = str(exc)
                log.warning("MQTT: cannot connect to %s:%s (%s); retrying in %.0f s", self.host, self.port, exc, delay)
                if self._stop.wait(delay):
                    return
                delay = min(15.0, delay * 2)
                continue
            delay = 1.0
            self._sock = sock
            self._last_rx = time.time()
            self.last_error = ""
            try:
                if self._subs:
                    self._send_subscribe(list(self._subs.items()))
            except OSError:
                self._drop("subscribe failed")
                continue
            self._connected.set()
            log.info("MQTT: connected to %s:%s", self.host, self.port)
            self._read_loop(sock)
            self._close(sock)
            if not self._stop.is_set():
                log.warning("MQTT: connection lost (%s); reconnecting", self.last_error or "closed")
                self._stop.wait(1.0)

    def _read_loop(self, sock) -> None:
        last_ping = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now - self._last_rx > self.keepalive * 1.5:
                self.last_error = "the broker stopped answering"
                return
            if now - last_ping >= self.keepalive / 2.0:
                try:
                    self._send(packet(PINGREQ, 0, b""))
                except OSError as exc:
                    self.last_error = str(exc)
                    return
                last_ping = now
            try:
                try:
                    head = sock.recv(1)                    # short timeout: only while waiting for a new packet
                except socket.timeout:
                    continue
                if not head:
                    raise ConnectionError("connection closed")
                sock.settimeout(30.0)                      # a packet that has started must be read whole
                try:
                    ptype, flags, body = read_packet(sock, head[0])
                finally:
                    sock.settimeout(1.0)
            except (OSError, MqttError, ConnectionError, ssl.SSLError) as exc:
                self.last_error = str(exc)
                return
            self._last_rx = time.time()
            if ptype == PUBACK and len(body) >= 2:
                ev = self._acks.get(struct.unpack("!H", body[:2])[0])
                if ev:
                    ev.set()
            elif ptype == PUBLISH:
                try:
                    topic, qos, pid, payload, retain, _ = parse_publish(flags, body)
                except (struct.error, UnicodeDecodeError):
                    self.last_error = "malformed message from the broker"
                    return
                ok = True
                if self.on_message:
                    try:
                        self.on_message(topic, payload, retain)
                    except Exception:
                        ok = False
                        log.exception("MQTT: handling a message on %s failed; it will not be acknowledged", topic)
                if qos == 1 and ok and pid is not None:
                    try:
                        self._send(packet(PUBACK, 0, struct.pack("!H", pid)))
                    except OSError as exc:
                        self.last_error = str(exc)
                        return
            # SUBACK / PINGRESP need no action
