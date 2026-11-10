"""Seat map: which seat is each tracked face sitting in?

The teacher marks the seats once on a still picture from the camera (dashboard > Seat map). After that,
every incident says *where* the student sits ("seat B4") and, if a seat has a roster student assigned, the
seat can also suggest the name when face matching is unsure.

Like face matching, the seat is only a SUGGESTION for the teacher to check against the video: students swap
seats, sit in the wrong one, or the camera moves. When the face match and the seat map disagree, the
dashboard says so.

Seats are stored as fractions of the picture (0..1), so they keep working if the analysis size changes.
The file is ``data/seatmap.json``.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import FaceObs

log = logging.getLogger("examguard.seatmap")

MAX_SEATS = 400
LABEL_RE = re.compile(r"^[^\s<>&\"'][^<>&\"']{0,19}$")      # short, printable, no markup characters


class SeatMapError(ValueError):
    pass


@dataclass
class Seat:
    label: str
    x: float                 # centre of the seat mark, 0..1 across the picture
    y: float                 # 0..1 down the picture
    camera: int = 0          # 0 = first camera, 1 = second camera
    student_ref: str = ""    # optional: roster student who is assigned to this seat

    def to_dict(self) -> dict:
        return asdict(self)


def row_letters(n: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA ..."""
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def validate_seats(raw: Sequence[dict], n_cameras: int = 2) -> List[Seat]:
    """Turn what the browser sent into clean Seat objects, or raise SeatMapError with a plain message."""
    if not isinstance(raw, (list, tuple)):
        raise SeatMapError("seats must be a list")
    if len(raw) > MAX_SEATS:
        raise SeatMapError(f"too many seats (limit {MAX_SEATS})")
    seats: List[Seat] = []
    seen = set()
    for i, r in enumerate(raw, start=1):
        if not isinstance(r, dict):
            raise SeatMapError(f"seat {i}: not an object")
        label = str(r.get("label", "")).strip()
        if not LABEL_RE.match(label):
            raise SeatMapError(f"seat {i}: the label must be 1-20 characters and not contain < > & or quotes")
        try:
            x, y = float(r["x"]), float(r["y"])
            cam = int(r.get("camera", 0))
        except (KeyError, TypeError, ValueError):
            raise SeatMapError(f"seat '{label}': it needs numeric x, y and camera")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) or math.isnan(x) or math.isnan(y):
            raise SeatMapError(f"seat '{label}': position must be inside the picture")
        if not 0 <= cam < n_cameras:
            raise SeatMapError(f"seat '{label}': unknown camera {cam}")
        key = (cam, label.lower())
        if key in seen:
            raise SeatMapError(f"two seats are called '{label}' on the same camera")
        seen.add(key)
        seats.append(Seat(label, round(x, 4), round(y, 4), cam, str(r.get("student_ref") or "")[:80]))
    return seats


def suggest_from_faces(faces: Iterable[Tuple[float, float, float, float]], camera: int = 0) -> List[Seat]:
    """Propose one seat per visible face, labelled by row (A, B, ...) and position in the row (1, 2, ...).

    ``faces`` are (cx, cy, w, h) as fractions of the picture. Rows are found by grouping faces whose
    vertical positions are within about half a face height. From a corner camera the rows are tilted,
    so this is a starting point for the teacher to correct, not the final answer.
    """
    items = sorted(faces, key=lambda f: f[1])
    if not items:
        return []
    heights = sorted(f[3] for f in items)
    gap = max(0.02, 0.6 * heights[len(heights) // 2])
    rows: List[List[Tuple[float, float, float, float]]] = []
    for f in items:
        if rows and abs(f[1] - (sum(p[1] for p in rows[-1]) / len(rows[-1]))) <= gap:
            rows[-1].append(f)
        else:
            rows.append([f])
    seats: List[Seat] = []
    for ri, row in enumerate(rows):
        for ci, f in enumerate(sorted(row, key=lambda p: p[0]), start=1):
            seats.append(Seat(f"{row_letters(ri)}{ci}", round(f[0], 4), round(f[1], 4), camera))
    return seats


class SeatMap:
    """The saved seats (thread-safe: the dashboard edits them while the monitor reads them)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._seats: List[Seat] = []
        if self.path and self.path.is_file():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._seats = validate_seats(data.get("seats", []))
        except (OSError, ValueError, SeatMapError) as exc:
            log.warning("seat map file %s could not be read (%s): starting with no seats", self.path, exc)
            self._seats = []

    # -------------------------------------------------------------- access
    def seats(self, camera: Optional[int] = None) -> List[Seat]:
        with self._lock:
            return [s for s in self._seats if camera is None or s.camera == camera]

    def __len__(self) -> int:
        with self._lock:
            return len(self._seats)

    def by_label(self, camera: int, label: str) -> Optional[Seat]:
        with self._lock:
            return next((s for s in self._seats if s.camera == camera and s.label == label), None)

    def replace(self, seats: List[Seat]) -> None:
        with self._lock:
            self._seats = list(seats)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"seats": [s.to_dict() for s in self._seats]}, indent=1), encoding="utf-8")
                tmp.replace(self.path)

    def clear_student(self, ref: str) -> None:
        """Forget a roster student (e.g. the roster changed): their seat stays but loses the name."""
        with self._lock:
            for s in self._seats:
                if s.student_ref == ref:
                    s.student_ref = ""


class SeatAssigner:
    """Per camera: decides which seat each tracked face is in, frame by frame.

    A face belongs to a seat when its centre is within ``radius`` face widths of the seat mark. Each seat
    holds at most one face, the nearest wins, and a face keeps its previous seat unless clearly closer to another.
    """
    STICKY = 0.6                  # distance to the previous seat counts as 60 % of its real value

    def __init__(self, seatmap: SeatMap, camera: int = 0, radius_faces: float = 1.6):
        self.map, self.camera, self.radius = seatmap, camera, radius_faces
        self._prev: Dict[int, str] = {}

    def assign(self, faces: Iterable[FaceObs], frame_w: int, frame_h: int) -> Dict[int, Seat]:
        seats = self.map.seats(self.camera)
        faces = list(faces)
        if not seats or not faces or frame_w <= 0 or frame_h <= 0:
            self._prev = {}
            return {}
        pairs = []
        for f in faces:
            cx, cy = f.center
            reach = self.radius * max(f.bbox[2], 1)
            for s in seats:
                d = math.hypot(cx - s.x * frame_w, cy - s.y * frame_h)
                if d <= reach:
                    score = d * (self.STICKY if self._prev.get(f.track_id) == s.label else 1.0)
                    pairs.append((score, f.track_id, s))
        pairs.sort(key=lambda p: p[0])
        out: Dict[int, Seat] = {}
        taken = set()
        for _, tid, seat in pairs:
            if tid in out or seat.label in taken:
                continue
            out[tid] = seat
            taken.add(seat.label)
        self._prev = {tid: s.label for tid, s in out.items()}
        return out
