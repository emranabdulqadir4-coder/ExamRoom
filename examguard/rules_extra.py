"""Extra behaviour rules: looking down, sideways eye glances, talking, and objects passed between students.

Like the head-turn rules these only ever raise *flags* for a teacher to review. They are heuristics:
each has plausible innocent causes (writing, thinking, whispering a question to the invigilator, reaching
for a pencil), which is why the human review step exists.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from .config import Config
from .models import Detection, Event, FaceObs
from .tracker import CentroidTracker

MIN_BASELINE_SAMPLES = 15
STALE_AFTER = 120.0
HAND_LABEL = "hand"


class _Ep:
    """One 'active for a while' episode (e.g. head down), tolerant of short dropouts."""

    def __init__(self, gap: float):
        self.gap, self.start, self.last, self.fired = gap, None, 0.0, False

    def step(self, ts: float, active: bool, hold: float):
        """Returns (fired_duration or None, (duration, had_fired) or None when the episode just ended)."""
        if active:
            self.last = ts
            if self.start is None:
                self.start, self.fired = ts, False
            if not self.fired and ts - self.start >= hold:
                self.fired = True
                return ts - self.start, None
            return None, None
        if self.start is not None and ts - self.last > self.gap:
            ended = (self.last - self.start, self.fired)
            self.start, self.fired = None, False
            return None, ended
        return None, None

    @property
    def open(self) -> bool:
        return self.start is not None


class _Series:
    """Recent values of one signal, with a cached median used as the student's own 'usual' level."""

    def __init__(self, window: float = 90.0):
        self.window, self.d, self._b, self._t = window, deque(), 0.0, -1e9

    def add(self, ts: float, v: float) -> None:
        self.d.append((ts, v))
        while self.d and ts - self.d[0][0] > self.window:
            self.d.popleft()

    def __len__(self) -> int:
        return len(self.d)

    def baseline(self, ts: float) -> float:
        if ts - self._t > 0.5 and self.d:
            self._b, self._t = statistics.median(v for _, v in self.d), ts
        return self._b


@dataclass
class _St:
    last_seen: float = 0.0
    face: Optional[FaceObs] = None
    series: Dict[str, _Series] = field(default_factory=dict)
    eps: Dict[str, _Ep] = field(default_factory=dict)
    counts: Dict[str, Deque[float]] = field(default_factory=dict)
    mouth: Deque[Tuple[float, float]] = field(default_factory=deque)
    cooldown: Dict[str, float] = field(default_factory=dict)
    note: Dict[str, str] = field(default_factory=dict)

    def s(self, name: str) -> _Series:
        return self.series.setdefault(name, _Series())

    def ep(self, name: str, gap: float) -> _Ep:
        return self.eps.setdefault(name, _Ep(gap))


class ExtraRules:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._st: Dict[int, _St] = {}
        self._hands = CentroidTracker(max_missing_seconds=0.8, match_factor=2.5)
        self._hand_hist: Dict[int, Deque[Tuple[float, Optional[int]]]] = {}
        self._hand_fired: Dict[int, float] = {}

    # ------------------------------------------------------------------ main
    def update(self, ts: float, faces: List[FaceObs], objects: Iterable[Detection]) -> List[Event]:
        cfg, events = self.cfg, []
        seen = set()
        for f in faces:
            seen.add(f.track_id)
            st = self._st.setdefault(f.track_id, _St())
            st.last_seen, st.face = ts, f
            if cfg.look_down_enabled:
                events += self._down(st, f, ts)
            if cfg.eye_gaze_enabled:
                events += self._gaze(st, f, ts)
            if cfg.talking_enabled:
                self._mouth(st, f, ts)
            st.s("yaw")                                    # make sure the series exists
            if f.yaw is not None:
                st.s("yaw").add(ts, f.yaw)
        for tid, st in self._st.items():                   # faces that vanished: let their episodes end
            if tid not in seen:
                for name, ep in list(st.eps.items()):
                    _, ended = ep.step(ts, False, 0)
                    if ended is not None:
                        events += self._after_end(st, tid, name, ts, *ended)
                if "talk" in st.eps and not st.eps["talk"].open:
                    st.mouth.clear()
        if cfg.talking_enabled:
            events += self._talking(ts, faces)
        if cfg.hands_enabled:
            events += self._passing(ts, faces, [d for d in objects if d.label == HAND_LABEL])
        self._prune(ts)
        return events

    # --------------------------------------------------- generic episode logic
    _KINDS = {
        "down": ("looking_down", "repeated_looking_down", "medium"),
        "gaze": ("eye_gaze", "repeated_eye_gaze", "medium"),
    }

    def _pattern(self, st: _St, tid: int, ts: float, name: str, active: bool, hold: float,
                 gap: float, note: str) -> List[Event]:
        if note:
            st.note[name] = note
        fire, ended = st.ep(name, gap).step(ts, active, hold)
        out: List[Event] = []
        if fire is not None:
            kind, _, sev = self._KINDS[name]
            ev = self._emit(st, ts, tid, kind, sev, 0.6, f"{st.note.get(name, '')} for {fire:.1f} s".strip())
            if ev:
                out.append(ev)
        if ended is not None:
            out += self._after_end(st, tid, name, ts, *ended)
        return out

    def _after_end(self, st: _St, tid: int, name: str, ts: float, duration: float, fired: bool) -> List[Event]:
        if name not in self._KINDS or fired:
            return []
        cfg = self.cfg
        min_len, count, window = ((cfg.down_min_seconds, cfg.down_count, cfg.down_window_seconds)
                                  if name == "down" else
                                  (cfg.gaze_min_seconds, cfg.gaze_count, cfg.gaze_window_seconds))
        if duration < min_len:
            return []
        q = st.counts.setdefault(name, deque())
        q.append(ts)
        while q and ts - q[0] > window:
            q.popleft()
        if len(q) < count:
            return []
        n = len(q)
        q.clear()
        _, kind, sev = self._KINDS[name]
        what = "dips below the usual posture" if name == "down" else "sideways eye glances"
        ev = self._emit(st, ts, tid, kind, sev, min(1.0, 0.5 + 0.1 * n), f"{n} {what} within {window:.0f} s")
        return [ev] if ev else []

    # ------------------------------------------------------------ looking down
    def _down(self, st: _St, f: FaceObs, ts: float) -> List[Event]:
        cfg = self.cfg
        if f.pitch is None and f.eyes_down is None:
            return []
        active, note, calibrating = False, "", False
        if f.pitch is not None:
            s = st.s("pitch")
            calibrating |= len(s) < MIN_BASELINE_SAMPLES
            rel = f.pitch - s.baseline(ts)
            s.add(ts, f.pitch)
            if rel >= cfg.down_threshold_deg:
                active, note = True, f"Head tilted down about {rel:.0f}° more than this student's usual posture"
        if f.eyes_down is not None:
            s = st.s("eyes_down")
            calibrating |= len(s) < MIN_BASELINE_SAMPLES
            rel_e = f.eyes_down - s.baseline(ts)
            s.add(ts, f.eyes_down)
            if rel_e >= cfg.eyes_down_threshold and not active:
                active, note = True, "Eyes looking down more than this student's usual"
        if calibrating:
            return []
        return self._pattern(st, f.track_id, ts, "down", active, cfg.down_seconds, 0.6, note)

    # --------------------------------------------------------------- eye gaze
    def _gaze(self, st: _St, f: FaceObs, ts: float) -> List[Event]:
        cfg = self.cfg
        if f.gaze_x is None:
            return []
        s = st.s("gaze")
        calibrating = len(s) < MIN_BASELINE_SAMPLES
        rel = f.gaze_x - s.baseline(ts)
        s.add(ts, f.gaze_x)
        if calibrating:
            return []
        ys = st.s("yaw")
        yaw_rel = (f.yaw - ys.baseline(ts)) if (f.yaw is not None and len(ys) >= MIN_BASELINE_SAMPLES) else 0.0
        head_forward = abs(yaw_rel) < 0.6 * cfg.yaw_threshold_deg      # a turned head is the head-turn rule's job
        active = abs(rel) >= cfg.gaze_threshold and head_forward
        side = "right" if rel > 0 else "left"
        return self._pattern(st, f.track_id, ts, "gaze", active, cfg.gaze_seconds, 0.4,
                             f"Eyes looking toward the {side} of the image while the head faces forward")

    # ----------------------------------------------------------------- talking
    def _mouth(self, st: _St, f: FaceObs, ts: float) -> None:
        cfg = self.cfg
        if f.mouth is None:
            return
        st.mouth.append((ts, f.mouth))
        while st.mouth and ts - st.mouth[0][0] > 2.5:
            st.mouth.popleft()
        vals = [v for _, v in st.mouth]
        moving = False
        if len(vals) >= 6:
            m = sum(vals) / len(vals)
            crossings = sum(1 for a, b in zip(vals, vals[1:]) if a < m <= b)
            moving = (max(vals) - min(vals)) >= cfg.mouth_amplitude and crossings >= cfg.mouth_cycles
        st.ep("talk", 0.8).step(ts, moving, cfg.talk_seconds)

    def _talking(self, ts: float, faces: List[FaceObs]) -> List[Event]:
        cfg = self.cfg
        talkers = [f for f in faces if (st := self._st.get(f.track_id)) and "talk" in st.eps
                   and st.eps["talk"].fired and st.eps["talk"].open]
        if len(talkers) < cfg.talk_min_students:
            return []
        parent = {f.track_id: f.track_id for f in talkers}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, a in enumerate(talkers):
            for b in talkers[i + 1:]:
                d = math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
                if d <= cfg.talk_neighbour_faces * max(a.bbox[2], b.bbox[2]):
                    parent[find(a.track_id)] = find(b.track_id)
        groups: Dict[int, List[FaceObs]] = {}
        for f in talkers:
            groups.setdefault(find(f.track_id), []).append(f)
        out: List[Event] = []
        for members in groups.values():
            if len(members) < cfg.talk_min_students:
                continue
            for f in members:
                others = ", ".join(f"Student {o.track_id}" for o in members if o is not f)
                st = self._st[f.track_id]
                dur = ts - (st.eps["talk"].start or ts)
                text = f"Mouth moving continuously for {dur:.1f} s" + (f", at the same time as {others}" if others else "")
                ev = self._emit(st, ts, f.track_id, "talking", "medium", min(1.0, 0.5 + 0.15 * len(members)), text)
                if ev:
                    out.append(ev)
        return out

    # ----------------------------------------------------------------- passing
    def _owner(self, hand: FaceObs, faces: List[FaceObs]) -> Optional[int]:
        hx, hy = hand.center
        best, best_d = None, 3.5
        for f in faces:
            fw = max(f.bbox[2], 1)
            dx, dy = (hx - f.center[0]) / fw, (hy - f.center[1]) / fw
            if dy < -1.0:                                # a hand far above the head is not at this student's desk
                continue
            d = math.hypot(dx, dy) * (2.0 if dy < 0 else 1.0)    # a face lower than the hand is probably not its owner
            if d < best_d:
                best, best_d = f.track_id, d
        return best

    def _passing(self, ts: float, faces: List[FaceObs], hands: List[Detection]) -> List[Event]:
        cfg = self.cfg
        tracked = self._hands.update(ts, [FaceObs(bbox=h.bbox) for h in hands])
        by_id = {f.track_id: f for f in faces}
        out: List[Event] = []
        live = set()
        for h in tracked:
            live.add(h.track_id)
            hist = self._hand_hist.setdefault(h.track_id, deque())
            hist.append((ts, self._owner(h, faces)))
            while hist and ts - hist[0][0] > cfg.pass_window_seconds + 1.0:
                hist.popleft()
            runs = self._runs(hist)
            if len(runs) < 2:
                continue
            (a, a0, a1), (b, b0, b1) = runs[-2], runs[-1]
            if (a == b or b1 < ts - 0.3 or b1 - b0 < cfg.pass_hold_seconds or a1 - a0 < cfg.pass_hold_seconds
                    or b0 - a1 > cfg.pass_window_seconds or self._hand_fired.get(h.track_id, -1e9) >= b0):
                continue
            fa, fb = by_id.get(a), by_id.get(b)
            st = self._st.get(a)
            if fa is None or fb is None or st is None:
                continue
            d = math.hypot(fa.center[0] - fb.center[0], fa.center[1] - fb.center[1])
            if d > cfg.talk_neighbour_faces * max(fa.bbox[2], fb.bbox[2]):
                continue
            self._hand_fired[h.track_id] = b1
            ev = self._emit(st, ts, a, "passing", "high", 0.8,
                            f"A hand moved from Student {a} to Student {b} (possible passing of a note or object)")
            if ev:
                out.append(ev)
        for hid in list(self._hand_hist):
            if hid not in live and not any(ts - t < 3.0 for t, _ in self._hand_hist[hid]):
                self._hand_hist.pop(hid, None)
                self._hand_fired.pop(hid, None)
        return out

    @staticmethod
    def _runs(hist) -> List[Tuple[int, float, float]]:
        runs: List[List] = []
        for t, owner in hist:
            if owner is None:
                continue
            if runs and runs[-1][0] == owner:
                runs[-1][2] = t
            else:
                runs.append([owner, t, t])
        return [tuple(r) for r in runs]

    # ----------------------------------------------------------------- helpers
    def _emit(self, st: _St, ts: float, tid: int, kind: str, severity: str, score: float,
              details: str) -> Optional[Event]:
        last = st.cooldown.get(kind)
        if last is not None and ts - last < self.cfg.cooldown_seconds:
            return None
        st.cooldown[kind] = ts
        return Event(ts=ts, track_id=tid, kind=kind, severity=severity, score=float(score), details=details,
                     bbox=st.face.bbox if st.face else None)

    def _prune(self, ts: float) -> None:
        for t in [t for t, s in self._st.items() if ts - s.last_seen > STALE_AFTER]:
            del self._st[t]
