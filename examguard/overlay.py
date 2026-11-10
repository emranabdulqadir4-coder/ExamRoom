"""Draw boxes and labels on frames (these annotated frames are also what goes into evidence clips)."""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

import cv2

from .models import Detection, FaceObs

GREEN = (90, 200, 90)
AMBER = (0, 170, 255)
RED = (70, 70, 235)
PURPLE = (220, 90, 170)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text: str, x: int, y: int, color) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, 0.5, 1)
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 5), (x + tw + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw(frame, faces: Iterable[FaceObs], objects: Iterable[Detection], engine,
         highlights: Dict[int, float], now: float, room: str, names: Optional[Dict[int, str]] = None,
         seats: Optional[Dict[int, str]] = None):
    """Annotate ``frame`` in place and return it."""
    for f in faces:
        x, y, w, h = f.bbox
        flagged = highlights.get(f.track_id, 0.0) > now
        turning = engine.is_turning(f.track_id)
        color = RED if flagged else AMBER if turning else GREEN
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3 if flagged else 2)
        nm = names.get(f.track_id) if names else None
        text = f"S{f.track_id}" + (f" [{seats[f.track_id]}]" if seats and f.track_id in seats and seats[f.track_id].isascii() else "") \
            + (f" {nm}" if nm and nm.isascii() else "")   # OpenCV cannot draw Arabic
        text += (f" {f.yaw:+.0f}°" if f.yaw is not None else "")
        if flagged:
            text += "  FLAGGED"
        _label(frame, text, x, y, color)
    for d in objects:
        x, y, w, h = d.bbox
        if d.label == "hand":                                  # hands: thin light box, no label clutter
            cv2.rectangle(frame, (x, y), (x + w, y + h), (230, 190, 90), 1)
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), PURPLE, 2)
        _label(frame, f"{d.label} {d.conf:.2f}", x, y, PURPLE)

    banner = f"{room}   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 24), (30, 30, 30), -1)
    cv2.putText(frame, banner, (8, 17), FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return frame


def no_signal_jpeg(width: int = 960, height: int = 540, text: str = "Waiting for camera...") -> bytes:
    import numpy as np
    img = np.full((height, width, 3), 28, np.uint8)
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.9, 2)
    cv2.putText(img, text, ((width - tw) // 2, height // 2), FONT, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
