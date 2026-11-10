"""Pure geometry for the extra face signals. Kept free of MediaPipe so it can be unit-tested.

`pts` is a sequence of (x, y) pixel positions for the 478 MediaPipe face landmarks (468 face + 10 iris).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

RIGHT_EYE = (33, 133)      # the two corners of one eye
LEFT_EYE = (263, 362)      # the two corners of the other eye
IRIS_A, IRIS_B = 468, 473  # iris centres
MOUTH_UP, MOUTH_DOWN, MOUTH_L, MOUTH_R = 13, 14, 78, 308


def gaze_ratio(pts: Sequence[Sequence[float]]) -> Optional[float]:
    """How far the irises sit from the middle of the eyes, as a fraction of eye width.

    Positive = toward the right of the image (the same convention as head yaw). Because it is measured
    inside the eye it barely changes when the whole head turns.
    """
    if len(pts) < 478:
        return None
    irises = [pts[IRIS_A], pts[IRIS_B]]
    vals = []
    for a, b in (RIGHT_EYE, LEFT_EYE):
        p, q = pts[a], pts[b]
        if q[0] < p[0]:
            p, q = q, p
        dx, dy = q[0] - p[0], q[1] - p[1]
        w = math.hypot(dx, dy)
        if w < 1e-6:
            return None
        mx, my = (p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0
        iris = min(irises, key=lambda i: math.hypot(i[0] - mx, i[1] - my))
        vals.append(((iris[0] - mx) * dx / w + (iris[1] - my) * dy / w) / w)
    return sum(vals) / 2.0


def mouth_openness(pts: Sequence[Sequence[float]]) -> Optional[float]:
    """Gap between the inner lips divided by mouth width (about 0 closed, 0.1-0.5 when open)."""
    if len(pts) < 400:
        return None
    gap = math.hypot(pts[MOUTH_UP][0] - pts[MOUTH_DOWN][0], pts[MOUTH_UP][1] - pts[MOUTH_DOWN][1])
    width = math.hypot(pts[MOUTH_L][0] - pts[MOUTH_R][0], pts[MOUTH_L][1] - pts[MOUTH_R][1])
    return gap / width if width > 1e-6 else None


def pitch_from_matrix(matrix, sign: float = 1.0) -> Optional[float]:
    """Head pitch in degrees (positive = tilted down) from MediaPipe's 4x4 facial transformation matrix.

    The matrix maps MediaPipe's canonical face (X right, Y up, Z toward the viewer) into camera space, so a
    rotation about X by +theta tips the top of the head toward the camera, i.e. looking DOWN. If your
    camera/convention turns out the other way round, set EXAMGUARD_PITCH_SIGN=-1 (see `tune.py --pose-check`).
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape[0] < 3 or m.shape[1] < 3:
        return None
    r = m[:3, :3]
    norms = np.linalg.norm(r, axis=0)
    if np.any(norms < 1e-9):
        return None
    r = r / norms                                       # remove any scale baked into the matrix
    return sign * math.degrees(math.atan2(r[2, 1], r[2, 2]))


def eyes_down_score(blendshapes: Dict[str, float]) -> Optional[float]:
    vals = [blendshapes[k] for k in ("eyeLookDownLeft", "eyeLookDownRight") if k in blendshapes]
    return sum(vals) / len(vals) if vals else None
