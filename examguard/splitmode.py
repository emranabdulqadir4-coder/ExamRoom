"""Edge box <-> server split over MQTT (Phase 2, optional).

    EDGE box (camera + AI, in the room)                       SERVER (teachers' dashboard)
    run.py --role edge                                        run.py --role server
      camera -> vision -> rules -> evidence clip                  review queue, roster + "Fail" marks,
      -> outbox (disk) -> MQTT broker ------------------------->  email alerts, retention, exam review
      status heartbeat, live picture, snapshot  --------------->  System check, live view, seat map
      <--------------------------------------------------------  seat map (retained message)

Rules this module keeps:
  * Nothing travels in the clear. Every payload is sealed with the shared key EXAMGUARD_MQTT_KEY (Fernet:
    encrypted AND authenticated), so someone who can reach the broker can neither read the video nor invent
    alerts. The broker's own password and TLS protect the connection on top of that.
  * An alert is never lost because the network or the server is down: the edge writes it to an on-disk outbox
    and resends until the broker acknowledges it (MQTT QoS 1). The server acknowledges only AFTER the alert is
    saved, and ignores repeats (each alert has a unique id), so delivery is at-least-once with no duplicates.
  * The edge keeps no footage once an alert is delivered; the server's vault and retention rules take over.
  * Human-in-the-loop is unchanged: everything arrives as a pending incident that a teacher must review.

HONEST STATUS: tested end to end against the in-process test broker in tests/minibroker.py, never against a real
Mosquitto broker, a real Raspberry Pi / Jetson, or two machines on a real network.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import socket
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken

from .config import Config
from .models import KIND_LABELS, Event
from .mqttlite import MqttClient
from .seatmap import SeatMap, SeatMapError, validate_seats

log = logging.getLogger("examguard.split")

WIRE_VERSION = 1
OFFLINE_AFTER = 20.0              # seconds without a heartbeat before the edge box counts as offline
MAX_CLOCK_SKEW = 5.0              # seconds the two computers' clocks may differ before System check warns
SEVERITIES = ("low", "medium", "high")


class WireError(ValueError):
    """A message that can never be accepted (wrong key, tampered, malformed). Retrying cannot fix it."""


def room_slug(room: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", room or "").strip("-").lower() or "room"


def seatmap_digest(seats) -> str:
    return hashlib.sha1(json.dumps([s.to_dict() for s in seats], sort_keys=True).encode()).hexdigest()[:12]


class Wire:
    """Seals and opens payloads with the shared key."""

    def __init__(self, key: str):
        key = (key or "").strip()
        if not key:
            raise ValueError(
                "EXAMGUARD_MQTT_KEY is required for --role edge / server (it encrypts everything sent through the broker). "
                "Make one with:  python scripts/make_key.py   and put the SAME value in .env on both computers.")
        try:
            self._f = Fernet(key.encode())
        except (ValueError, TypeError) as exc:
            raise ValueError("EXAMGUARD_MQTT_KEY is not a valid key: make one with python scripts/make_key.py") from exc

    def seal(self, data: bytes) -> bytes:
        return self._f.encrypt(data)

    def open(self, token: bytes) -> bytes:
        try:
            return self._f.decrypt(token)
        except InvalidToken as exc:
            raise WireError("could not be decrypted: EXAMGUARD_MQTT_KEY differs between the computers, "
                            "or the message was tampered with") from exc

    def seal_json(self, obj) -> bytes:
        return self.seal(json.dumps(obj, separators=(",", ":")).encode())

    def open_json(self, token: bytes):
        try:
            return json.loads(self.open(token))
        except ValueError as exc:
            raise WireError("not valid JSON") from exc


class Topics:
    def __init__(self, prefix: str, room: str):
        self.base = f"{prefix.strip('/') or 'examguard'}/{room_slug(room)}/"
        self.incident = self.base + "incident"
        self.status = self.base + "status"
        self.seatmap = self.base + "seatmap"

    def live(self, i: int) -> str:
        return f"{self.base}live/{i}"

    def snap(self, i: int) -> str:
        return f"{self.base}snap/{i}"


def make_client(cfg: Config, client_id: str, on_message, clean_session: bool) -> MqttClient:
    if not cfg.mqtt_host:
        raise ValueError("EXAMGUARD_MQTT_HOST is required for --role edge / server (the broker's address)")
    if not cfg.mqtt_tls and cfg.mqtt_host not in ("localhost", "127.0.0.1", "::1"):
        log.warning("MQTT without TLS: ExamGuard's own key encrypts every payload, but the broker password is "
                    "sent unprotected. Set EXAMGUARD_MQTT_TLS=true when your broker supports it")
    return MqttClient(cfg.mqtt_host, cfg.mqtt_port, client_id, cfg.mqtt_user, cfg.mqtt_password, cfg.mqtt_tls,
                      cfg.mqtt_cafile, clean_session=clean_session, on_message=on_message)


# -------------------------------------------------------------------- events
def event_to_dict(ev: Event) -> dict:
    d = asdict(ev)
    if d.get("bbox") is not None:
        d["bbox"] = list(d["bbox"])
    return d


def event_from_dict(d) -> Event:
    """Strict: anything odd is rejected instead of guessed."""
    if not isinstance(d, dict):
        raise WireError("event is not an object")
    try:
        ev = Event(ts=float(d["ts"]), track_id=int(d["track_id"]), kind=str(d["kind"]), severity=str(d["severity"]),
                   score=float(d["score"]), details=str(d["details"])[:2000])
    except (KeyError, TypeError, ValueError) as exc:
        raise WireError(f"event is incomplete ({exc})") from exc
    if ev.kind not in KIND_LABELS:
        raise WireError(f"unknown rule '{ev.kind}'")
    if ev.severity not in SEVERITIES:
        raise WireError("unknown severity")
    if not 0.0 <= ev.score <= 1.0 or not 0 < ev.ts < 4e9:
        raise WireError("score or time out of range")
    bbox = d.get("bbox")
    if bbox is not None:
        try:
            ev.bbox = tuple(float(x) for x in bbox)
            if len(ev.bbox) != 4:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise WireError("bad bounding box") from exc
    for name in ("student_name", "student_ref", "seat", "seat_student", "id_source"):
        v = d.get(name)
        if v is not None:
            if not isinstance(v, str) or len(v) > 200:
                raise WireError(f"bad {name}")
            setattr(ev, name, v)
    if d.get("match_score") is not None:
        try:
            ev.match_score = float(d["match_score"])
        except (TypeError, ValueError) as exc:
            raise WireError("bad match score") from exc
    ev.camera = str(d.get("camera") or "")[:60]
    return ev


def _b64(data: Optional[bytes]) -> Optional[str]:
    return base64.b64encode(data).decode() if data else None


def _unb64(s) -> Optional[bytes]:
    if not s:
        return None
    try:
        return base64.b64decode(s, validate=True)
    except (ValueError, TypeError) as exc:
        raise WireError("bad attachment") from exc


def build_incident_message(wire: Wire, node: str, room: str, event: Event,
                           clip: Optional[bytes], thumb: Optional[bytes]) -> Tuple[str, bytes]:
    msg_id = secrets.token_hex(16)
    body = {"v": WIRE_VERSION, "id": msg_id, "node": node, "room": room, "event": event_to_dict(event),
            "clip": _b64(clip), "thumb": _b64(thumb), "sent": time.time()}
    return msg_id, wire.seal_json(body)


# ======================================================================= EDGE
class EdgeLink:
    """Runs on the box in the room: outbox + sender, heartbeat, live picture, snapshot, seat map from the server."""

    def __init__(self, cfg: Config, wire: Wire, vault, seatmap: SeatMap):
        self.cfg, self.wire, self.vault, self.seatmap = cfg, wire, vault, seatmap
        self.node = cfg.node_name.strip() or socket.gethostname()
        self.topics = Topics(cfg.mqtt_topic_prefix, cfg.room_name)
        self.outbox = cfg.data_path / "outbox"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.monitors: List = []
        self.dropped = 0
        self.sent = 0
        self.ack_timeout = 15.0                     # how long to wait for the broker's acknowledgement before retrying
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        self.client = make_client(cfg, f"examguard-edge-{room_slug(cfg.room_name)}-{room_slug(self.node)}",
                                  self._on_message, clean_session=True)
        self.client.subscribe(self.topics.seatmap, 1)

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.client.start()
        for target, name in ((self._send_loop, "outbox"), (self._periodic_loop, "heartbeat")):
            t = threading.Thread(target=target, name="edge-" + name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        for t in self._threads:
            t.join(timeout=5)
        self.client.stop()

    # ------------------------------------------------------------- outbox
    def pending(self) -> List[Path]:
        return sorted(self.outbox.glob("*.msg"))

    def submit_incident(self, event: Event, clip_name: Optional[str], thumb_name: Optional[str]) -> str:
        """Called when an evidence clip is finished. Seals the alert into the outbox and removes the local footage."""
        clip = thumb = None
        try:
            if clip_name:
                clip = self.vault.get(clip_name)
            if thumb_name:
                thumb = self.vault.get(thumb_name)
        except Exception:
            log.exception("edge: could not read the evidence it just recorded; sending the alert without it")
        msg_id, data = build_incident_message(self.wire, self.node, self.cfg.room_name, event, clip, thumb)
        name = f"{time.time_ns():020d}-{msg_id}.msg"
        tmp = self.outbox / (name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, self.outbox / name)
        for n in (clip_name, thumb_name):           # delivered footage must not pile up on the box
            if n:
                try:
                    self.vault.delete(n)
                except Exception:
                    pass
        self._enforce_cap()
        self._wake.set()
        return msg_id

    def _enforce_cap(self) -> None:
        cap = int(self.cfg.outbox_max_mb * 1024 * 1024)
        files = self.pending()
        total = sum(f.stat().st_size for f in files if f.exists())
        while files and total > cap:
            oldest = files.pop(0)
            try:
                total -= oldest.stat().st_size
                oldest.unlink()
            except OSError:
                continue
            self.dropped += 1
            log.error("edge: the outbox is over %s MB (the server has been unreachable for too long): "
                      "the OLDEST undelivered alert was dropped", self.cfg.outbox_max_mb)

    def _send_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            files = self.pending()
            if not files:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            try:
                data = files[0].read_bytes()
            except OSError:
                continue
            if self.client.publish(self.topics.incident, data, qos=1, timeout=self.ack_timeout):
                try:
                    files[0].unlink()
                except OSError:
                    pass
                self.sent += 1
                backoff = 1.0
                continue
            self._wake.wait(backoff)
            self._wake.clear()
            backoff = min(30.0, backoff * 2)

    # ------------------------------------------------- heartbeat + pictures
    def status_message(self) -> dict:
        return {"v": WIRE_VERSION, "node": self.node, "room": self.cfg.room_name, "now": time.time(),
                "profile": self.cfg.profile, "outbox": len(self.pending()), "dropped": self.dropped,
                "sent": self.sent, "seat_digest": seatmap_digest(self.seatmap.seats()),
                "cameras": [m.status() for m in self.monitors]}

    def _periodic_loop(self) -> None:
        last_status = last_live = last_snap = 0.0
        while not self._stop.wait(0.25):
            now = time.time()
            try:
                if now - last_status >= self.cfg.heartbeat_seconds:
                    last_status = now
                    self.client.publish(self.topics.status, self.wire.seal_json(self.status_message()), qos=0)
                if now - last_live >= self.cfg.live_seconds:
                    last_live = now
                    for i, m in enumerate(self.monitors):
                        jpeg, _ = m.wait_frame(-1, 0.05)
                        if jpeg:
                            self.client.publish(self.topics.live(i), self.wire.seal(jpeg), qos=0)
                if now - last_snap >= max(2.0, self.cfg.live_seconds * 5):
                    last_snap = now
                    for i, m in enumerate(self.monitors):
                        jpeg = m.snapshot_jpeg()
                        if jpeg:
                            self.client.publish(self.topics.snap(i), self.wire.seal_json(
                                {"jpeg": _b64(jpeg), "faces": [list(f) for f in m.snapshot_faces()]}), qos=0)
            except Exception:
                log.exception("edge: heartbeat failed")

    # ---------------------------------------------------- seat map from server
    def _on_message(self, topic: str, payload: bytes, retained: bool) -> None:
        if topic != self.topics.seatmap:
            return
        try:
            data = self.wire.open_json(payload)
            seats = validate_seats(data.get("seats", []), n_cameras=max(1, len(self.monitors)))
        except (WireError, SeatMapError, AttributeError) as exc:
            log.warning("edge: ignoring a seat map from the server (%s)", exc)
            return
        if seatmap_digest(seats) == seatmap_digest(self.seatmap.seats()):
            return
        self.seatmap.replace(seats)
        log.info("edge: seat map updated from the server (%d seats)", len(seats))

    def info(self) -> dict:
        return {"node": self.node, "connected": self.client.connected, "outbox": len(self.pending()),
                "dropped": self.dropped, "sent": self.sent, "last_error": self.client.last_error}


# ====================================================================== SERVER
class RemoteFeed:
    """What the server knows about the edge box, built from its heartbeats and pictures."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cond = threading.Condition()
        self.status: Optional[dict] = None
        self.status_at = 0.0
        self.skew = 0.0
        self.nodes: Dict[str, float] = {}
        self.monitors: List["RemoteMonitor"] = [RemoteMonitor(self, 0)]

    def update_status(self, d: dict) -> None:
        cams = d.get("cameras")
        if not isinstance(cams, list) or not all(isinstance(c, dict) for c in cams):
            raise WireError("bad status")
        now = time.time()
        with self.cond:
            self.status, self.status_at = d, now
            self.skew = float(d.get("now", now)) - now
            self.nodes[str(d.get("node", "?"))] = now
            while len(self.monitors) < min(len(cams), 2):
                self.monitors.append(RemoteMonitor(self, len(self.monitors)))
            self.nodes = {k: v for k, v in self.nodes.items() if now - v < 120}

    def monitor(self, i: int) -> Optional["RemoteMonitor"]:
        return self.monitors[i] if 0 <= i < len(self.monitors) else None


class RemoteMonitor:
    """Stands in for a local Monitor on the server, so the dashboard code does not care where the camera is."""

    def __init__(self, feed: RemoteFeed, index: int):
        self.feed, self.camera_index = feed, index
        self._jpeg: Optional[bytes] = None
        self._frame_id = 0
        self._snap: Optional[bytes] = None
        self._faces: List[Tuple[float, float, float, float]] = []

    @property
    def camera_name(self) -> str:
        d = self.feed.status
        cams = d["cameras"] if d else []
        return (cams[self.camera_index].get("camera") or "") if self.camera_index < len(cams) else ""

    # the Monitor interface used by the dashboard
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_live(self, jpeg: bytes) -> None:
        with self.feed.cond:
            self._jpeg, self._frame_id = jpeg, self._frame_id + 1
            self.feed.cond.notify_all()

    def set_snap(self, jpeg: bytes, faces) -> None:
        self._snap = jpeg
        self._faces = [tuple(float(x) for x in f) for f in faces if isinstance(f, (list, tuple)) and len(f) == 4]

    def wait_frame(self, last_id: int, timeout: float = 5.0) -> Tuple[Optional[bytes], int]:
        with self.feed.cond:
            if self._frame_id == last_id:
                self.feed.cond.wait(timeout)
            if self._frame_id == last_id:
                return None, last_id
            return self._jpeg, self._frame_id

    def snapshot_jpeg(self) -> Optional[bytes]:
        return self._snap

    def snapshot_faces(self):
        return list(self._faces)

    def status(self) -> dict:
        f = self.feed
        base = {"state": "offline", "source": "edge box: nothing received yet", "camera": self.camera_name,
                "source_state": "", "fps": 0.0, "students": 0, "face_health": {}, "last_frame_age": None,
                "phone_detector": False, "hand_detector": False, "analyzer": "edge", "uptime": 0, "error": "",
                "remote": True, "edge_age": None, "node": ""}
        d = f.status
        if d is None:
            return base
        cams = d.get("cameras") or []
        st = {**base, **(cams[self.camera_index] if self.camera_index < len(cams) else {})}
        age = time.time() - f.status_at
        st.update(remote=True, edge_age=round(age, 1), node=str(d.get("node", "")),
                  source=f"edge box {d.get('node', '?')}: {st.get('source', '')}")
        if age > OFFLINE_AFTER:
            st.update(state="offline", source_state="", fps=0.0)
        elif st.get("last_frame_age") is not None:
            st["last_frame_age"] = round(st["last_frame_age"] + age, 1)
        return st


class ServerLink:
    """Runs on the teachers' computer: receives alerts, status and pictures; sends the seat map."""

    def __init__(self, cfg: Config, wire: Wire, store, vault, hub, notifier, seatmap: SeatMap, feed: RemoteFeed):
        self.cfg, self.wire, self.store, self.vault, self.hub = cfg, wire, store, vault, hub
        self.notifier, self.seatmap, self.feed = notifier, seatmap, feed
        self.topics = Topics(cfg.mqtt_topic_prefix, cfg.room_name)
        self.received = 0
        self.duplicates = 0
        self.rejected = 0
        self._last_seat_push = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # persistent session: the broker keeps alerts for us while this computer is off or restarting
        self.client = make_client(cfg, f"examguard-server-{room_slug(cfg.room_name)}", self._on_message,
                                  clean_session=False)
        self.client.subscribe(self.topics.incident, 1)
        for t in (self.topics.status, self.topics.base + "live/+", self.topics.base + "snap/+"):
            self.client.subscribe(t, 0)

    def start(self) -> None:
        self.client.start()
        self._thread = threading.Thread(target=self._seat_sync_loop, name="seat-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.client.stop()

    # ------------------------------------------------------------ incoming
    def _on_message(self, topic: str, payload: bytes, retained: bool) -> None:
        if topic == self.topics.incident:
            try:
                self.ingest_incident(payload)
            except WireError as exc:                    # can never succeed: record it and acknowledge (drop)
                self.rejected += 1
                log.error("server: rejected an alert from the edge box (%s)", exc)
                self.store.audit("system", "edge_message_rejected", None, str(exc)[:200])
            return                                      # any other error propagates: no ack, the broker resends
        try:
            if topic == self.topics.status:
                self.feed.update_status(self.wire.open_json(payload))
            elif topic.startswith(self.topics.base + "live/"):
                m = self.feed.monitor(int(topic.rsplit("/", 1)[1]))
                if m:
                    m.set_live(self.wire.open(payload))
            elif topic.startswith(self.topics.base + "snap/"):
                m = self.feed.monitor(int(topic.rsplit("/", 1)[1]))
                if m:
                    d = self.wire.open_json(payload)
                    jpeg = _unb64(d.get("jpeg"))
                    if jpeg:
                        m.set_snap(jpeg, d.get("faces") or [])
        except (WireError, ValueError, AttributeError, TypeError) as exc:
            log.debug("server: ignoring a status/picture message (%s)", exc)

    def ingest_incident(self, payload: bytes) -> Optional[dict]:
        """Save one alert from the edge. Returns the incident, or None when it was a repeat."""
        msg = self.wire.open_json(payload)
        if not isinstance(msg, dict) or msg.get("v") != WIRE_VERSION:
            raise WireError("unsupported message version")
        rid = msg.get("id")
        if not isinstance(rid, str) or not re.fullmatch(r"[0-9a-f]{32}", rid):
            raise WireError("bad alert id")
        event = event_from_dict(msg.get("event"))
        clip, thumb = _unb64(msg.get("clip")), _unb64(msg.get("thumb"))
        if self.store.has_remote_id(rid):
            self.duplicates += 1
            return None                                  # already saved (the edge did not hear our acknowledgement)
        clip_name = self.vault.put(f"clip-r{rid}.mp4", clip) if clip else None
        thumb_name = self.vault.put(f"thumb-r{rid}.jpg", thumb) if thumb else None
        incident = self.store.add_incident(event, self.cfg.room_name, clip_name, thumb_name, remote_id=rid)
        self.store.audit("system", "edge_received", incident["id"],
                         f"from {str(msg.get('node', '?'))[:40]}, sent {time.strftime('%H:%M:%S', time.localtime(float(msg.get('sent') or 0)))}")
        self.received += 1
        self.hub.publish("incident", incident)
        self.notifier.notify(incident)
        return incident

    # ----------------------------------------------------------- seat map out
    def publish_seatmap(self, seats=None) -> bool:
        seats = self.seatmap.seats() if seats is None else seats
        data = self.wire.seal_json({"seats": [s.to_dict() for s in seats]})
        self._last_seat_push = time.time()
        return self.client.publish(self.topics.seatmap, data, qos=1, retain=True, timeout=5.0)

    def _seat_sync_loop(self) -> None:
        """Self-healing: if the edge box reports a different seat map than ours (for example the broker lost its
        retained message when it restarted), send ours again. Only when we have seats and it is not brand new."""
        while not self._stop.wait(5.0):
            try:
                d = self.feed.status
                if not d or not self.client.connected or time.time() - self._last_seat_push < 30.0:
                    continue
                if time.time() - self.feed.status_at > OFFLINE_AFTER:
                    continue
                seats = self.seatmap.seats()
                if seats and d.get("seat_digest") != seatmap_digest(seats):
                    self.publish_seatmap(seats)
            except Exception:
                log.exception("server: seat map sync failed")

    def info(self) -> dict:
        f = self.feed
        age = None if not f.status else time.time() - f.status_at          # unrounded: the offline test compares it
        d = f.status or {}
        return {"connected": self.client.connected, "last_error": self.client.last_error, "edge_age": age,
                "nodes": sorted(f.nodes), "skew": round(f.skew, 1), "outbox": d.get("outbox"),
                "dropped": d.get("dropped"), "received": self.received, "duplicates": self.duplicates,
                "rejected": self.rejected, "edge_profile": d.get("profile")}
