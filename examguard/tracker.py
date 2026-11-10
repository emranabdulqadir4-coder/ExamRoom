"""Keeps a stable id for each student across frames (nearest-centroid matching).

Students stay seated, so matching a face to the closest previously seen face is
enough. A track survives a short disappearance (e.g. a head turned far away).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from .models import FaceObs


@dataclass
class _Track:
    id: int
    center: tuple
    width: float
    last_seen: float


class CentroidTracker:
    def __init__(self, max_missing_seconds: float = 2.0, match_factor: float = 1.0):
        self.max_missing = max_missing_seconds
        self.match_factor = match_factor
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def update(self, ts: float, faces: List[FaceObs]) -> List[FaceObs]:
        self._tracks = {i: t for i, t in self._tracks.items() if ts - t.last_seen <= self.max_missing}

        pairs = []
        for t in self._tracks.values():
            for j, f in enumerate(faces):
                cx, cy = f.center
                dist = math.hypot(cx - t.center[0], cy - t.center[1])
                if dist <= self.match_factor * max(t.width, f.bbox[2]):
                    pairs.append((dist, t.id, j))
        pairs.sort()

        used_tracks, used_faces = set(), set()
        for _, tid, j in pairs:
            if tid in used_tracks or j in used_faces:
                continue
            used_tracks.add(tid)
            used_faces.add(j)
            faces[j].track_id = tid
            t = self._tracks[tid]
            t.center, t.width, t.last_seen = faces[j].center, float(faces[j].bbox[2]), ts

        for j, f in enumerate(faces):
            if j in used_faces:
                continue
            tid = self._next_id
            self._next_id += 1
            f.track_id = tid
            self._tracks[tid] = _Track(tid, f.center, float(f.bbox[2]), ts)
        return faces
