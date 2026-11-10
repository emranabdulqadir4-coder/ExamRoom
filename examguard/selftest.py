"""Guided self-test: record YOURSELF doing known behaviours, and get the labels file for free.

No classmates and no equipment needed. The program tells you what to do and when ("turn your head now").
Because it knows when each instruction was shown, it writes the ``labels.csv`` that ``scripts/evaluate.py``
needs, so you do not have to watch the video and note times by hand.

It also asks you to do harmless things (stretch, look up, write normally). Those are NOT labelled, so any
alert during them is counted as a false alarm: that is how you measure how often normal behaviour is flagged.

HONEST LIMITS: one person, one seat, one camera position. Talking only works with
``EXAMGUARD_TALK_MIN_STUDENTS=1`` (it normally needs two neighbours talking) and passing a note needs two people,
so it is not tested here. The numbers describe YOUR face in YOUR room, not a real classroom with a corner camera.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

LEAD_IN = 20.0          # seconds of normal sitting first, so ExamGuard learns your usual posture
GET_READY = 3.0         # warning before each action
REST = 7.0              # back to normal between actions (lets the 20 s alert cool-down pass partly)
COOLDOWN_GAP = 12.0     # extra quiet time after a labelled action so one alert cannot spill into the next label


@dataclass(frozen=True)
class Step:
    kind: Optional[str]       # evaluation label kind, or None for a harmless "decoy" action
    title: str
    instruction: str
    seconds: float            # how long to keep doing it


BEHAVIOURS = {
    "head_turn": Step("head_turn", "Turn your head",
                      "Turn your head to the SIDE and look at the wall, as if peeking at a neighbour's paper. Hold it.", 5.0),
    "looking_down": Step("looking_down", "Look down",
                         "Tilt your head DOWN as if reading a hidden sheet in your lap. Hold it.", 6.0),
    "eye_gaze": Step("eye_gaze", "Look sideways with your eyes",
                     "Keep your head facing the screen, but look to the SIDE with your eyes only. Hold it.", 4.0),
    "phone": Step("phone", "Show a phone",
                  "Hold your phone up in front of you so the camera can see it. Hold it.", 4.0),
    "talking": Step("talking", "Talk",
                    "Keep talking or whispering continuously (count out loud).", 7.0),
}
DECOYS = [
    Step(None, "Stretch", "Stretch your arms and neck. (Normal behaviour: should NOT be flagged.)", 5.0),
    Step(None, "Look up and think", "Look up at the ceiling as if thinking. (Normal: should NOT be flagged.)", 5.0),
    Step(None, "Write normally", "Look down and write on paper with natural small head movements. (Normal.)", 10.0),
    Step(None, "Quick glance at the clock", "Glance to one side for about one second, then back. (Normal.)", 1.5),
]


def build_script(repeats: int = 3, phone: bool = False, talking: bool = False) -> List[Step]:
    """Each round does every behaviour once, with a harmless decoy action between them."""
    names = ["head_turn", "looking_down", "eye_gaze"] + (["phone"] if phone else []) + (["talking"] if talking else [])
    steps: List[Step] = []
    d = 0
    for _ in range(max(1, repeats)):
        for n in names:
            steps.append(BEHAVIOURS[n])
            steps.append(DECOYS[d % len(DECOYS)])
            d += 1
    return steps


@dataclass(frozen=True)
class Slot:
    step: Step
    ready_at: float      # the "get ready" warning starts
    start: float         # the action starts (label start)
    end: float           # the action ends (label end)
    next_at: float       # when the next slot's warning may begin


def build_timeline(steps: Sequence[Step], lead_in: float = LEAD_IN) -> Tuple[List[Slot], float]:
    slots: List[Slot] = []
    t = lead_in
    for st in steps:
        start = t + GET_READY
        end = start + st.seconds
        quiet = COOLDOWN_GAP if st.kind else REST
        slots.append(Slot(st, t, start, end, end + quiet))
        t = end + quiet
    return slots, t


def labels_for(slots: Sequence[Slot]) -> List[Tuple[float, float, str]]:
    """(start, end, kind) for the labelled actions only; decoys are intentionally left out."""
    return [(round(s.start, 2), round(s.end, 2), s.step.kind) for s in slots if s.step.kind]


def write_labels(slots: Sequence[Slot], path: str) -> int:
    rows = labels_for(slots)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "kind"])
        w.writerows(rows)
    return len(rows)


def prompt_at(slots: Sequence[Slot], total: float, t: float, lead_in: float = LEAD_IN) -> dict:
    """What to show the person at video time ``t``."""
    if t >= total:
        return {"phase": "done", "title": "Finished", "text": "All done. Stay still for a moment.", "left": 0.0, "n": len(slots), "i": len(slots)}
    if t < lead_in:
        return {"phase": "lead_in", "title": "Sit normally", "text": "Sit and write as in an exam. Do nothing special.",
                "left": lead_in - t, "n": len(slots), "i": 0}
    for i, s in enumerate(slots, start=1):
        if t < s.ready_at:
            break
        if t < s.start:
            return {"phase": "ready", "title": "Get ready: " + s.step.title, "text": s.step.instruction, "left": s.start - t,
                    "n": len(slots), "i": i}
        if t < s.end:
            return {"phase": "go", "title": "NOW: " + s.step.title, "text": s.step.instruction, "left": s.end - t,
                    "n": len(slots), "i": i}
        if t < s.next_at:
            return {"phase": "rest", "title": "Back to normal", "text": "Sit and write normally.", "left": s.next_at - t,
                    "n": len(slots), "i": i}
    return {"phase": "rest", "title": "Back to normal", "text": "Sit and write normally.", "left": 0.0, "n": len(slots), "i": len(slots)}


def record_session(source, out_video: str, slots: Sequence[Slot], total: float, fps: float = 15.0,
                   show: Optional[Callable[[object, float], bool]] = None,
                   clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                   max_width: int = 1280) -> float:
    """Record ``total`` seconds from ``source`` into an MP4 whose frame count follows the wall clock exactly,
    so video time equals timeline time even if the camera is slower than ``fps`` (frames are repeated).

    ``show(frame, t)`` may draw the instructions in a window; returning False stops early.
    Returns the recorded length in seconds."""
    import cv2

    frame = None
    wait_until = clock() + 15.0
    while frame is None:
        frame, _ = source.read(0.5)
        if frame is None and clock() > wait_until:
            raise RuntimeError("the camera gave no picture for 15 seconds")
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    else:
        w, h = w // 2 * 2, h // 2 * 2
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write the video file {out_video}")
    latest, written, t0 = frame, 0, clock()
    try:
        while True:
            t = clock() - t0
            if t >= total:
                break
            fr, _ = source.read(0.05)
            if fr is not None:
                latest = fr
            due = int(t * fps) + 1
            while written < due:
                writer.write(cv2.resize(latest, (w, h)) if latest.shape[1] != w or latest.shape[0] != h else latest)
                written += 1
            if show is not None and show(latest, t) is False:
                break
            sleep(0.005)
    finally:
        writer.release()
    return written / fps


def draw_prompt(frame, info: dict):
    """A copy of the frame with big instructions on it (for the preview window only, never the recording)."""
    import cv2
    img = frame.copy()
    h, w = img.shape[:2]
    colour = {"go": (60, 170, 60), "ready": (0, 150, 230), "lead_in": (110, 110, 110), "rest": (110, 110, 110),
              "done": (60, 170, 60)}.get(info["phase"], (110, 110, 110))
    cv2.rectangle(img, (0, 0), (w, 92), colour, -1)
    cv2.putText(img, info["title"], (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    text = info["text"]
    scale = 0.55 if len(text) > 70 else 0.7
    cv2.putText(img, text[:120], (14, 72), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{info['left']:.0f}s", (w - 90, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    if info["n"]:
        cv2.putText(img, f"step {info['i']}/{info['n']}   (press Q to stop)", (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img
