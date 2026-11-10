"""Measure how a particular camera and room behave, so the alert thresholds can be set sensibly.

Run `python scripts/tune.py` while people sit normally for a minute: it reports each student's usual
head angle, how far normal movement strays from it, and how many pixels wide the faces are.
"""
from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from typing import Dict, List, Optional

import cv2

from .tracker import CentroidTracker


def _pct(values: List[float], p: float) -> float:
    s = sorted(values)
    return s[int(p * (len(s) - 1))]


class Collector:
    def __init__(self):
        self.frames = 0
        self.yaws: Dict[int, List[float]] = defaultdict(list)
        self.widths: Dict[int, List[float]] = defaultdict(list)
        self.pitches: Dict[int, List[float]] = defaultdict(list)
        self.gazes: Dict[int, List[float]] = defaultdict(list)
        self.mouths: Dict[int, List[float]] = defaultdict(list)

    def add(self, faces) -> None:
        self.frames += 1
        for f in faces:
            self.widths[f.track_id].append(f.bbox[2])
            if f.yaw is not None:
                self.yaws[f.track_id].append(f.yaw)
            for store, v in ((self.pitches, f.pitch), (self.gazes, f.gaze_x), (self.mouths, f.mouth)):
                if v is not None:
                    store[f.track_id].append(v)

    def pitch_mean(self) -> Optional[float]:
        """Average head pitch of the face seen most often (used by the pose check)."""
        if not self.pitches:
            return None
        tid = max(self.pitches, key=lambda t: len(self.pitches[t]))
        return statistics.fmean(self.pitches[tid])

    @staticmethod
    def _spread(values: List[float]) -> Optional[float]:
        if len(values) < 20:
            return None
        base = statistics.median(values)
        return _pct([abs(v - base) for v in values], 0.99)

    def per_student(self, min_samples: int = 20) -> List[Dict]:
        out = []
        for tid, ys in sorted(self.yaws.items()):
            if len(ys) < min_samples:
                continue
            base = statistics.median(ys)
            dev = [abs(y - base) for y in ys]
            out.append({"track_id": tid, "samples": len(ys), "baseline": base,
                        "p99": _pct(dev, 0.99), "max": max(dev),
                        "face_px": statistics.median(self.widths[tid]),
                        "pitch_p99": self._spread(self.pitches.get(tid, [])),
                        "gaze_p99": self._spread(self.gazes.get(tid, [])),
                        "mouth_open": (sum(1 for v in self.mouths[tid] if v > 0.08) / len(self.mouths[tid])
                                       if self.mouths.get(tid) else None)})
        return out

    def report(self, threshold: float, min_face_px: float = 40.0, cfg=None) -> str:
        rows = self.per_student()
        if not rows:
            return ("Not enough data: faces were rarely or never detected. Check the camera view, the lighting "
                    "and that people face the camera, then run it again.")
        lines = [f"Measured over {self.frames} frames. Alert threshold is currently {threshold:.0f} degrees.", ""]
        for r in rows:
            lines.append(f"  Student {r['track_id']}: usual angle {r['baseline']:+.0f}, normal movement up to "
                         f"{r['p99']:.0f} degrees from it (99% of the time, worst {r['max']:.0f}), "
                         f"face {r['face_px']:.0f} px wide, {r['samples']} readings")
        worst = max(r["p99"] for r in rows)
        lines.append("")
        if worst + 10 >= threshold:
            suggested = math.ceil((worst + 15) / 5) * 5
            lines.append(f"WARNING: normal movement reaches {worst:.0f} degrees, close to the {threshold:.0f} degree "
                         f"threshold: expect false alerts. Raise EXAMGUARD_YAW_THRESHOLD_DEG to about {suggested}.")
        else:
            lines.append(f"Normal movement ({worst:.0f} degrees at most) is well below the threshold: no false alerts expected from it.")
        if cfg is not None:
            lines += self._extra_report(rows, cfg)
        small = [r for r in rows if r["face_px"] < min_face_px]
        if small:
            lines.append(f"WARNING: {len(small)} face(s) are under {min_face_px:.0f} px wide; readings for them are unreliable "
                         f"(move the camera closer, zoom in, or use a higher resolution).")
        lines += ["", "This only measures NORMAL sitting. To check that real glances are caught, turn your head "
                      "on purpose toward a neighbour and watch the angle shown on the live video."]
        return "\n".join(lines)


    @staticmethod
    def _extra_report(rows, cfg) -> List[str]:
        out = []
        pitch = [r["pitch_p99"] for r in rows if r["pitch_p99"] is not None]
        gaze = [r["gaze_p99"] for r in rows if r["gaze_p99"] is not None]
        mouth = [r["mouth_open"] for r in rows if r["mouth_open"] is not None]
        if pitch:
            w = max(pitch)
            out.append(f"Head up/down: normal posture changes reach {w:.0f} degrees; 'looking down' fires at "
                       f"{cfg.down_threshold_deg:.0f}. " + (
                           f"TOO CLOSE: raise EXAMGUARD_DOWN_THRESHOLD_DEG to about {math.ceil((w + 10) / 5) * 5}."
                           if w + 5 >= cfg.down_threshold_deg else "OK."))
        else:
            out.append("Head up/down: not measured (no pitch readings).")
        if gaze:
            w = max(gaze)
            out.append(f"Eye position: normal movement reaches {w:.2f}; 'eye glance' fires at {cfg.gaze_threshold:.2f}. " + (
                f"TOO CLOSE: raise EXAMGUARD_GAZE_THRESHOLD to about {math.ceil((w + 0.04) * 20) / 20:.2f}."
                if w + 0.03 >= cfg.gaze_threshold else "OK."))
        else:
            out.append("Eye position: not measured.")
        if mouth:
            w = max(mouth)
            out.append(f"Mouths: open in up to {w:.0%} of readings. " + (
                "That is high for a quiet room (talking, chewing or yawning?). Expect false 'talking' flags."
                if w > 0.25 else "OK for a quiet room."))
        return out


def decide_pitch_sign(neutral: Optional[float], down: Optional[float], current: float):
    """From 'looking straight' vs 'looking down' readings: is EXAMGUARD_PITCH_SIGN right? -> (new_sign|None, message)"""
    if neutral is None or down is None:
        return None, "No face was detected during the check. Make sure your face is clearly visible and try again."
    delta = down - neutral
    if abs(delta) < 5:
        return None, (f"Inconclusive: the pitch hardly changed ({delta:+.1f} degrees). Tilt your head down "
                      f"clearly (chin toward chest) and try again.")
    if delta > 0:
        return current, f"Correct: looking down raised the reading by {delta:.0f} degrees. Keep EXAMGUARD_PITCH_SIGN={current:g}."
    new = -current
    return new, (f"Looking down LOWERED the reading by {-delta:.0f} degrees, so the direction is reversed. "
                 f"Set EXAMGUARD_PITCH_SIGN={new:g} in your .env file.")


def pose_check(source, analyzer, cfg, say=print, wait=input, seconds: float = 3.0) -> str:
    wait("Sit in front of the camera. Look STRAIGHT at it, then press Enter and hold still... ")
    neutral = run(source, analyzer, cfg.max_width, seconds).pitch_mean()
    wait("Now tilt your head DOWN (chin toward chest, as if reading your desk). Press Enter, then hold it... ")
    down = run(source, analyzer, cfg.max_width, seconds).pitch_mean()
    new, message = decide_pitch_sign(neutral, down, cfg.pitch_sign)
    say(message)
    return message


def run(source, analyzer, max_width: int, seconds: float) -> Collector:
    """Read frames for `seconds`, analysing each one. `source`/`analyzer` are the same objects the app uses."""
    col, tracker = Collector(), CentroidTracker()
    end, last_ms = time.time() + seconds, 0
    while time.time() < end:
        frame, _ = source.read(0.5)
        if frame is None:
            if getattr(source, "state", "") == "ended":
                break
            continue
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
        now = time.time()
        last_ms = max(last_ms + 1, int(now * 1000))
        col.add(tracker.update(now, analyzer.analyze(frame, last_ms)))
    return col
