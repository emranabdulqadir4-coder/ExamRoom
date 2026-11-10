"""Measure how well the detection rules work on a recorded video that YOU have labelled.

You watch a mock exam recording and note when real events happen (start, end, kind). ExamGuard replays the
same video through the same pipeline and compares its alerts with your notes. The result is the number the
proposal needs: for each behaviour, how many real events were caught (recall), how many alerts were real
(precision), and how many false alarms per hour a teacher would have to dismiss.

Labels file (CSV with a header row)::

    start,end,kind
    0:42,0:48,head_turn
    1:10,1:14,phone
    2:05,2:12,talking

Times are seconds or m:ss, measured from the start of the video. ``kind`` is one of the names below.
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2

from .config import Config
from .models import Detection, Event, FaceObs
from .rules import RuleEngine
from .tracker import CentroidTracker

# An alert of any kind in a group counts for a label of that group ("repeated" alerts belong with their base rule).
GROUPS = {
    "head_turn": "head_turn", "repeated_glancing": "head_turn",
    "phone": "phone",
    "looking_down": "looking_down", "repeated_looking_down": "looking_down",
    "eye_gaze": "eye_gaze", "repeated_eye_gaze": "eye_gaze",
    "talking": "talking",
    "passing": "passing",
}
LABEL_KINDS = tuple(dict.fromkeys(GROUPS.values()))
OBJECT_TTL = 1.0
MIN_LABELS_FOR_TRUST = 20

Step = Tuple[float, List[FaceObs], List[Detection]]


def group_of(kind: str) -> str:
    return GROUPS.get(kind, kind)


# ---------------------------------------------------------------- labels
@dataclass
class Label:
    start: float
    end: float
    kind: str


def parse_time(text: str) -> float:
    text = str(text).strip()
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        secs = 0.0
        for p in parts:
            secs = secs * 60 + p
        return secs
    return float(text)


def load_labels(path: str) -> List[Label]:
    labels: List[Label] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no label rows (expected a header 'start,end,kind').")
    for n, row in enumerate(rows, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        try:
            start, end, kind = parse_time(row["start"]), parse_time(row["end"]), row["kind"].lower()
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path} line {n}: need numeric start, end and a kind ({exc})") from exc
        if kind not in LABEL_KINDS:
            raise ValueError(f"{path} line {n}: unknown kind '{kind}'. Use one of: {', '.join(LABEL_KINDS)}")
        if end < start:
            raise ValueError(f"{path} line {n}: end is before start")
        labels.append(Label(start, end, kind))
    return labels


# ---------------------------------------------------------------- replay
def replay(cfg: Config, steps: Iterable[Step]) -> Tuple[List[Event], float]:
    """Run observations through the tracker and the rules exactly as the live monitor does.

    Returns (alerts, duration in seconds). Timestamps are video time, so a 1-hour video is processed as fast as
    the models allow, not in real time."""
    tracker, rules = CentroidTracker(), RuleEngine(cfg)
    events: List[Event] = []
    last_ts = 0.0
    held: List[Detection] = []
    held_ts = -1e9
    for ts, faces, objects in steps:
        last_ts = ts
        if objects:
            held, held_ts = list(objects), ts
        elif ts - held_ts > OBJECT_TTL:
            held = []
        tracked = tracker.update(ts, faces)
        events += rules.update(ts, tracked, held)
    return events, last_ts


def video_steps(path: str, cfg: Config, analyzer, detector=None,
                max_seconds: Optional[float] = None) -> Iterator[Step]:
    """Read a video file at cfg.process_fps (by video time) and yield analysed observations."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / max(cfg.process_fps, 1)))
    idx, last_ms = 0, 0
    try:
        while True:
            if not cap.grab():
                break
            ts = idx / fps
            idx += 1
            if (idx - 1) % stride:
                continue
            if max_seconds is not None and ts > max_seconds:
                break
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w > cfg.max_width:
                frame = cv2.resize(frame, (cfg.max_width, int(h * cfg.max_width / w)), interpolation=cv2.INTER_AREA)
            last_ms = max(last_ms + 1, int(ts * 1000))
            faces = analyzer.analyze(frame, last_ms)
            objects = detector.detect(frame) if detector is not None else []
            yield ts, faces, objects
    finally:
        cap.release()


# ---------------------------------------------------------------- scoring
@dataclass
class GroupScore:
    kind: str
    labels: int = 0
    detected: int = 0
    alerts: int = 0
    correct_alerts: int = 0
    delays: List[float] = field(default_factory=list)
    missed: List[Label] = field(default_factory=list)
    false_alarm_times: List[float] = field(default_factory=list)

    @property
    def recall(self) -> Optional[float]:
        return self.detected / self.labels if self.labels else None

    @property
    def precision(self) -> Optional[float]:
        return self.correct_alerts / self.alerts if self.alerts else None

    @property
    def false_alarms(self) -> int:
        return self.alerts - self.correct_alerts


@dataclass
class Report:
    duration: float
    tolerance: float
    groups: Dict[str, GroupScore]

    @property
    def false_alarms(self) -> int:
        return sum(g.false_alarms for g in self.groups.values())

    @property
    def false_alarms_per_hour(self) -> Optional[float]:
        return self.false_alarms / (self.duration / 3600.0) if self.duration > 0 else None


def score(events: Sequence[Event], labels: Sequence[Label], duration: float, tolerance: float = 3.0,
          kinds: Optional[Sequence[str]] = None) -> Report:
    """Match alerts to labels. An alert is correct if a label of the same behaviour covers its time (+/- tolerance).
    Several alerts for one label are not false alarms; the label counts as detected once."""
    wanted = list(kinds) if kinds else list(LABEL_KINDS)
    groups = {k: GroupScore(k) for k in wanted}
    for lab in labels:
        if lab.kind in groups:
            groups[lab.kind].labels += 1
    for ev in events:
        g = groups.get(group_of(ev.kind))
        if g is None:
            continue
        g.alerts += 1
        if any(l.kind == g.kind and l.start - tolerance <= ev.ts <= l.end + tolerance for l in labels):
            g.correct_alerts += 1
        else:
            g.false_alarm_times.append(ev.ts)
    for lab in labels:
        g = groups.get(lab.kind)
        if g is None:
            continue
        hits = sorted(e.ts for e in events
                      if group_of(e.kind) == lab.kind and lab.start - tolerance <= e.ts <= lab.end + tolerance)
        if hits:
            g.detected += 1
            g.delays.append(hits[0] - lab.start)
        else:
            g.missed.append(lab)
    return Report(duration, tolerance, groups)


def _fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.0%}"


def format_report(rep: Report) -> str:
    lines = [f"Video length: {_fmt_t(rep.duration)}   (an alert counts if it falls within "
             f"{rep.tolerance:g} s of a labelled event)", ""]
    head = f"{'behaviour':<14}{'labelled':>9}{'caught':>8}{'recall':>8}{'alerts':>8}{'correct':>9}{'precision':>11}{'delay':>9}"
    lines += [head, "-" * len(head)]
    for g in rep.groups.values():
        if not g.labels and not g.alerts:
            continue
        delay = f"{statistics.median(g.delays):.1f} s" if g.delays else "-"
        lines.append(f"{g.kind:<14}{g.labels:>9}{g.detected:>8}{_pct(g.recall):>8}{g.alerts:>8}"
                     f"{g.correct_alerts:>9}{_pct(g.precision):>11}{delay:>9}")
    fa = rep.false_alarms_per_hour
    lines += ["", f"False alarms: {rep.false_alarms} in total"
                  + (f", about {fa:.1f} per hour of exam" if fa is not None else "")
                  + ". Each one is something a teacher must dismiss."]
    for g in rep.groups.values():
        if g.missed:
            lines.append(f"  missed {g.kind}: " + ", ".join(f"{_fmt_t(m.start)}-{_fmt_t(m.end)}" for m in g.missed[:12]))
        if g.false_alarm_times:
            lines.append(f"  false {g.kind} alerts at: " + ", ".join(_fmt_t(t) for t in g.false_alarm_times[:12]))
    small = [g.kind for g in rep.groups.values() if 0 < g.labels < MIN_LABELS_FOR_TRUST]
    lines += ["", "How far to trust this:"]
    if small:
        lines.append(f"  - Fewer than {MIN_LABELS_FOR_TRUST} labelled events for: {', '.join(small)}. "
                     f"Percentages from so few events are rough. Label more, or use more videos.")
    lines.append("  - Anything the system flagged that you did not label counts as a false alarm, so label everything "
                 "real that you want judged (or pass --kinds to judge only some behaviours).")
    lines.append("  - Results describe this room, camera and these people only.")
    return "\n".join(lines)


def events_to_csv(events: Sequence[Event], path: str) -> None:
    """Write the alerts so a person can scrub the video to each time and decide whether it was real."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time", "seconds", "kind", "student_track", "details"])
        for e in sorted(events, key=lambda e: e.ts):
            w.writerow([_fmt_t(e.ts), f"{e.ts:.1f}", e.kind, e.track_id, e.details])


# ---------------------------------------------------------------- self-check on the simulator
DEMO_PLAN = [  # (seconds from start, scenario, student)
    (25, "head_turn", 1), (40, "phone", 5), (55, "look_down", 2), (70, "eye_gaze", 4),
    (85, "talking", 1), (105, "passing", 4), (120, "glancing", 6), (140, "head_turn", 3),
]


def demo_run(cfg: Config, fps: float = 10.0) -> Tuple[List[Event], float, List[Label]]:
    """Replay a scripted simulated exam whose true events are known: checks the evaluation tool itself."""
    import time

    from .demo import SCENARIO_SECONDS, DemoWorld
    world = DemoWorld(cfg, auto=False)
    base = time.time()
    labels: List[Label] = []
    kind_of = {"glancing": "head_turn", "look_down": "looking_down", "eye_gaze": "eye_gaze"}
    for at, kind, student in DEMO_PLAN:
        world.trigger(kind, student, delay=at)
        labels.append(Label(at, at + SCENARIO_SECONDS[kind], kind_of.get(kind, kind)))
    duration = max(l.end for l in labels) + 10.0

    def steps() -> Iterator[Step]:
        for i in range(int(duration * fps)):
            faces, objs = world.observations_at(base + i / fps)
            yield i / fps, faces, objs
    events, dur = replay(cfg, steps())
    return events, dur, labels
