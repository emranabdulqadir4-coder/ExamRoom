"""The monitoring loop: camera -> faces -> tracker -> rules -> annotated frame -> evidence clip."""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2

from . import overlay
from .clips import ClipRecorder
from .config import Config
from .health import face_health
from .models import Detection
from .rules import RuleEngine
from .tracker import CentroidTracker

log = logging.getLogger("examguard.monitor")

OBJECT_TTL = 1.0          # a phone detection stays valid this long (the detector runs slower than the video)
FLAG_HIGHLIGHT = 4.0      # seconds a flagged student keeps a red box


class Monitor:
    def __init__(self, cfg: Config, source, analyzer, detector, recorder: ClipRecorder,
                 source_label: str = "camera", identifier=None, camera_name: str = "",
                 seat_assigner=None, roster=None, camera_index: int = 0):
        self.cfg, self.source, self.analyzer, self.detector, self.recorder = cfg, source, analyzer, detector, recorder
        self.identifier = identifier                 # optional: suggests roster names for faces
        self.camera_name = camera_name               # "" for a one-camera room
        self.camera_index = camera_index
        self.seat_assigner = seat_assigner           # optional: which seat is each face in
        self.roster = roster                         # optional: lets a seat suggest a name
        self._raw = None                             # latest clean (unannotated) frame, for the seat-map picture
        self._seat_faces: List[Tuple[float, float, float, float]] = []
        self.source_label = source_label
        self.tracker = CentroidTracker()
        self.rules = RuleEngine(cfg)
        self._highlight: Dict[int, float] = {}

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._obj_thread: Optional[threading.Thread] = None
        self._obj_wake = threading.Event()
        self._obj_frame = None
        self._objects: Tuple[float, List[Detection]] = (0.0, [])
        self._obj_lock = threading.Lock()

        self._cond = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._frame_id = 0
        self._last_ms = 0
        self._started = time.time()
        self._last_frame_ts = 0.0
        self._fps = 0.0
        self._faces = 0
        self._health = face_health([], cfg.min_face_px)
        self._errors = 0
        self._state = "starting"
        self._last_error = ""

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="monitor", daemon=True)
        self._thread.start()
        if self.detector is not None:
            self._obj_thread = threading.Thread(target=self._object_loop, name="objects", daemon=True)
            self._obj_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._obj_wake.set()
        for t in (self._thread, self._obj_thread):
            if t:
                t.join(timeout=5)
        try:
            self.source.close()
        except Exception:
            pass
        self.recorder.close(flush=True)
        close = getattr(self.analyzer, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass

    # ---------------------------------------------------------- live access
    def wait_frame(self, last_id: int, timeout: float = 5.0) -> Tuple[Optional[bytes], int]:
        with self._cond:
            if self._frame_id == last_id:
                self._cond.wait(timeout)
            if self._frame_id == last_id:
                return None, last_id
            return self._jpeg, self._frame_id

    def snapshot_jpeg(self) -> Optional[bytes]:
        """A clean picture (no boxes) of the latest frame, used to mark seats."""
        raw = self._raw
        if raw is None:
            return None
        ok, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def snapshot_faces(self) -> List[Tuple[float, float, float, float]]:
        """Visible faces as (cx, cy, w, h) fractions of the picture, to propose seats."""
        return list(self._seat_faces)

    def status(self) -> dict:
        age = time.time() - self._last_frame_ts if self._last_frame_ts else None
        state = self._state
        if state == "running" and age is not None and age > 3:
            state = "stalled"
        return {
            "state": state,
            "source": self.source_label,
            "camera": self.camera_name,
            "source_state": getattr(self.source, "state", ""),
            "fps": round(self._fps, 1),
            "students": self._faces,
            "face_health": self._health,
            "last_frame_age": None if age is None else round(age, 1),
            "phone_detector": self.detector is not None and "phone" in getattr(self.detector, "names", ["phone"]),
            "hand_detector": self.detector is not None and "hands" in getattr(self.detector, "names", []),
            "analyzer": getattr(self.analyzer, "name", "unknown"),
            "uptime": int(time.time() - self._started),
            "error": self._last_error,
        }

    # ----------------------------------------------------------------- loop
    def _run(self) -> None:
        period = 1.0 / self.cfg.process_fps
        last_fid = -1
        while not self._stop.is_set():
            t0 = time.time()
            self.recorder.tick(t0)
            try:
                frame, fid = self.source.read(timeout=0.5)
                if frame is None or fid == last_fid:
                    if self._state != "running":
                        self._state = "waiting for video"
                    continue
                last_fid = fid
                self._process(frame, t0)
                self._state, self._errors = "running", 0
            except Exception as exc:
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._state = "error"
                log.exception("frame processing failed")
                self._stop.wait(min(2.0, 0.2 * self._errors))
                continue
            dt = time.time() - t0
            self._fps = 0.8 * self._fps + 0.2 * (1.0 / max(dt, 1e-3)) if self._fps else 1.0 / max(dt, 1e-3)
            if not getattr(self.source, "paced", False) and dt < period:
                self._stop.wait(period - dt)

    def _process(self, frame, now: float) -> None:
        h, w = frame.shape[:2]
        if w > self.cfg.max_width:
            frame = cv2.resize(frame, (self.cfg.max_width, int(h * self.cfg.max_width / w)),
                               interpolation=cv2.INTER_AREA)
        ts_ms = max(self._last_ms + 1, int(now * 1000))
        self._last_ms = ts_ms

        faces = self.analyzer.analyze(frame, ts_ms)
        self._offer_object_frame(frame)
        faces = self.tracker.update(now, faces)
        fh, fw = frame.shape[:2]
        self._raw = frame.copy()
        self._seat_faces = [((f.center[0]) / fw, (f.center[1]) / fh, f.bbox[2] / fw, f.bbox[3] / fh) for f in faces]
        seat_of = self.seat_assigner.assign(faces, fw, fh) if self.seat_assigner is not None else {}
        self._health = face_health([f.bbox[2] for f in faces], self.cfg.min_face_px)
        objects = self._fresh_objects(now)
        if self.identifier is not None:
            try:
                self.identifier.observe(now, frame, faces)
            except Exception:                        # name matching must never stop monitoring
                log.exception("face matching failed")
        events = self.rules.update(now, faces, objects)
        for ev in events:
            self._enrich(ev, seat_of)

        for ev in events:
            self._highlight[ev.track_id] = now + FLAG_HIGHLIGHT
        names = self.identifier.names() if self.identifier is not None else None
        annotated = overlay.draw(frame, faces, objects, self.rules, self._highlight, now,
                                 self.cfg.room_name + (f" - {self.camera_name}" if self.camera_name else ""), names,
                                 {t: s.label for t, s in seat_of.items()})
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        jpeg = buf.tobytes()

        self.recorder.push_frame(now, jpeg)       # first, so the flagged frame is inside the pre-roll
        for ev in events:
            log.info("potential incident: %s - %s (%s)", ev.label, ev.kind, ev.details)
            self.recorder.trigger(ev)

        with self._cond:
            self._jpeg, self._frame_id = jpeg, self._frame_id + 1
            self._faces, self._last_frame_ts = len(faces), time.time()
            self._cond.notify_all()

    def _enrich(self, ev, seat_of) -> None:
        """Add the suggested name (face matching, then seat map) and the seat to a new event. Suggestions only."""
        ev.camera = self.camera_name
        seat = seat_of.get(ev.track_id)
        seat_student = None
        if seat is not None:
            ev.seat = seat.label
            if seat.student_ref and self.roster is not None:
                seat_student = self.roster.get(seat.student_ref)
                ev.seat_student = seat_student.name if seat_student else None
        ident = None
        if self.identifier is not None:
            try:
                ident = self.identifier.identity(ev.track_id)
            except Exception:                         # name matching must never stop monitoring
                log.exception("identity lookup failed")
        if ident:
            ev.student_name, ev.student_ref, ev.match_score, ev.id_source = ident.name, ident.ref, ident.score, "face"
        elif seat_student is not None:
            ev.student_name, ev.student_ref, ev.id_source = seat_student.name, seat_student.ref, "seat"

    # ------------------------------------------------------ object detection
    def _offer_object_frame(self, frame) -> None:
        if self.detector is not None and not self._obj_wake.is_set():
            self._obj_frame = frame.copy()          # copy: the original is drawn on afterwards
            self._obj_wake.set()

    def _object_loop(self) -> None:
        while not self._stop.is_set():
            if not self._obj_wake.wait(0.5):
                continue
            frame = self._obj_frame
            try:
                dets = self.detector.detect(frame)
                with self._obj_lock:
                    self._objects = (time.time(), dets)
            except Exception:
                log.exception("object detection failed")
                self._stop.wait(1.0)
            self._obj_wake.clear()

    def _fresh_objects(self, now: float) -> List[Detection]:
        with self._obj_lock:
            t, dets = self._objects
        return list(dets) if now - t <= OBJECT_TTL else []
