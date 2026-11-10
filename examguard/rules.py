"""Behaviour rules: turn per-frame observations into *potential* incident events.

The engine is pure logic (time is passed in), so it is easy to unit-test and tune.
It never decides that anyone cheated; it only emits Events for a teacher to review.

Rules
-----
head_turn          head turned away from the student's own baseline by at least
                   ``yaw_threshold_deg`` for ``turn_seconds`` in a row (side-glancing at a neighbour)
repeated_glancing  ``glance_count`` shorter glances within ``glance_window_seconds``
phone              a phone is detected next to a student for ``phone_seconds``
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from .config import Config
from .models import Detection, Event, FaceObs
from .rules_extra import ExtraRules

PHONE_LABEL = "cell phone"


@dataclass
class _Student:
    last_seen: float = 0.0
    last_face: Optional[FaceObs] = None
    samples: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=1500))
    last_rel: float = 0.0
    # head turn
    turn_start: Optional[float] = None
    turn_fired: bool = False
    turn_sign: int = 0
    last_turn_ts: float = 0.0
    glances: Deque[float] = field(default_factory=deque)
    # phone
    phone_since: Optional[float] = None
    phone_seen: float = 0.0
    phone_fired: bool = False
    phone_conf: float = 0.0
    # cooldowns
    last_event: Dict[str, float] = field(default_factory=dict)


class RuleEngine:
    BASELINE_WINDOW = 90.0        # seconds of history used for the baseline angle
    MIN_BASELINE_SAMPLES = 15     # frames needed before turn detection starts for a student
    HYSTERESIS = 0.8              # a turn only ends when the angle falls below 80% of the threshold
    PHONE_HOLD = 0.8              # seconds without a phone detection before its timer resets
    STALE_AFTER = 120.0

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.extra = ExtraRules(cfg)
        self._s: Dict[int, _Student] = {}

    # ------------------------------------------------------------------ public
    def update(self, ts: float, faces: List[FaceObs], objects: Iterable[Detection] = ()) -> List[Event]:
        events: List[Event] = []
        visible = set()
        for f in faces:
            visible.add(f.track_id)
            st = self._s.setdefault(f.track_id, _Student())
            st.last_seen = ts
            st.last_face = f
            events += self._head(st, f.track_id, f, ts)
        for tid, st in self._s.items():
            if tid not in visible and st.turn_start is not None:
                events += self._lost_while_turning(st, tid, ts)
        objects = list(objects)
        events += self._phone(ts, faces, objects)
        events += self.extra.update(ts, faces, objects)
        self._prune(ts)
        return events

    def is_turning(self, track_id: int) -> bool:
        st = self._s.get(track_id)
        return bool(st and st.turn_start is not None)

    def last_face(self, track_id: int) -> Optional[FaceObs]:
        st = self._s.get(track_id)
        return st.last_face if st else None

    # ------------------------------------------------------------- head turn
    def _baseline(self, st: _Student) -> float:
        if not self.cfg.use_baseline:
            return 0.0
        return statistics.median(y for _, y in st.samples) if st.samples else 0.0

    def _head(self, st: _Student, tid: int, face: FaceObs, ts: float) -> List[Event]:
        if face.yaw is None:
            return []
        cfg = self.cfg
        calibrating = cfg.use_baseline and len(st.samples) < self.MIN_BASELINE_SAMPLES
        rel = face.yaw - self._baseline(st)
        st.last_rel = rel
        st.samples.append((ts, face.yaw))
        while st.samples and ts - st.samples[0][0] > self.BASELINE_WINDOW:
            st.samples.popleft()
        if calibrating:
            return []

        thr = cfg.yaw_threshold_deg
        turning = abs(rel) >= thr
        if st.turn_start is not None:
            same_side = (rel > 0) == (st.turn_sign > 0)
            turning = abs(rel) >= thr * self.HYSTERESIS and same_side

        if turning:
            if st.turn_start is None:
                st.turn_start, st.turn_fired = ts, False
                st.turn_sign = 1 if rel > 0 else -1
            st.last_turn_ts = ts
            if not st.turn_fired and ts - st.turn_start >= cfg.turn_seconds:
                st.turn_fired = True
                ev = self._turn_event(st, tid, ts, rel)
                return [ev] if ev else []
            return []
        if st.turn_start is not None:
            return self._end_turn(st, tid, ts)
        return []

    def _turn_event(self, st: _Student, tid: int, ts: float, rel: float) -> Optional[Event]:
        side = "right" if st.turn_sign > 0 else "left"
        dur = ts - st.turn_start
        return self._emit(
            st, ts, tid, "head_turn", "medium", min(1.0, abs(rel) / 70.0),
            f"Head turned about {abs(rel):.0f}° toward the {side} of the image for {dur:.1f} s",
        )

    def _end_turn(self, st: _Student, tid: int, ts: float) -> List[Event]:
        cfg = self.cfg
        duration = st.last_turn_ts - (st.turn_start or st.last_turn_ts)
        fired = st.turn_fired
        st.turn_start, st.turn_fired = None, False
        if fired or duration < cfg.glance_min_seconds:
            return []
        st.glances.append(ts)
        while st.glances and ts - st.glances[0] > cfg.glance_window_seconds:
            st.glances.popleft()
        if len(st.glances) >= cfg.glance_count:
            n = len(st.glances)
            st.glances.clear()
            ev = self._emit(
                st, ts, tid, "repeated_glancing", "medium", min(1.0, 0.5 + 0.1 * n),
                f"{n} side glances within {cfg.glance_window_seconds:.0f} s",
            )
            return [ev] if ev else []
        return []

    def _lost_while_turning(self, st: _Student, tid: int, ts: float) -> List[Event]:
        """The face disappeared mid-turn (turned so far the landmarks were lost)."""
        if ts - st.last_seen <= self.cfg.turn_gap_seconds:
            st.last_turn_ts = ts
            if not st.turn_fired and ts - st.turn_start >= self.cfg.turn_seconds:
                st.turn_fired = True
                ev = self._turn_event(st, tid, ts, st.last_rel)
                return [ev] if ev else []
            return []
        return self._end_turn(st, tid, ts)

    # ----------------------------------------------------------------- phone
    @staticmethod
    def _nearest_face(det: Detection, faces: List[FaceObs]) -> Optional[FaceObs]:
        px, py = det.center
        best, best_d = None, 3.5
        for f in faces:
            fx, fy = f.center
            fw = max(f.bbox[2], 1)
            dx, dy = (px - fx) / fw, (py - fy) / fw
            if dy < -1.2:                    # a phone far above the head is not "this student's"
                continue
            d = math.hypot(dx, dy)
            if d < best_d:
                best, best_d = f, d
        return best

    def _phone(self, ts: float, faces: List[FaceObs], objects: List[Detection]) -> List[Event]:
        cfg = self.cfg
        out: List[Event] = []
        hits: Dict[int, float] = {}
        for d in objects:
            if d.label != PHONE_LABEL:
                continue
            f = self._nearest_face(d, faces)
            if f is not None:
                hits[f.track_id] = max(hits.get(f.track_id, 0.0), d.conf)
        for tid, conf in hits.items():
            st = self._s[tid]
            st.phone_seen, st.phone_conf = ts, conf
            if st.phone_since is None:
                st.phone_since, st.phone_fired = ts, False
            if not st.phone_fired and ts - st.phone_since >= cfg.phone_seconds:
                st.phone_fired = True
                ev = self._emit(
                    st, ts, tid, "phone", "high", float(min(1.0, conf)),
                    f"Phone visible near the student for {ts - st.phone_since:.1f} s (confidence {conf:.0%})",
                )
                if ev:
                    out.append(ev)
        for tid, st in self._s.items():
            if tid not in hits and st.phone_since is not None and ts - st.phone_seen > self.PHONE_HOLD:
                st.phone_since, st.phone_fired = None, False
        return out

    # --------------------------------------------------------------- helpers
    def _emit(self, st: _Student, ts: float, tid: int, kind: str, severity: str,
              score: float, details: str) -> Optional[Event]:
        last = st.last_event.get(kind)
        if last is not None and ts - last < self.cfg.cooldown_seconds:
            return None
        st.last_event[kind] = ts
        bbox = st.last_face.bbox if st.last_face else None
        return Event(ts=ts, track_id=tid, kind=kind, severity=severity,
                     score=float(score), details=details, bbox=bbox)

    def _prune(self, ts: float) -> None:
        stale = [t for t, s in self._s.items()
                 if ts - s.last_seen > self.STALE_AFTER and s.turn_start is None]
        for t in stale:
            del self._s[t]
