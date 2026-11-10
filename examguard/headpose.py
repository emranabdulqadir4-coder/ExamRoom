"""Head-turn (yaw) estimation from three face-mesh landmarks.

Why not solvePnP? A generic 3D face model plus PnP gives noisy, ambiguous angles
for small faces in a wide classroom view. This closed-form estimate is stable,
cheap, easy to explain, and unaffected by head roll:

    offset = horizontal position of the nose tip relative to the midpoint of the
             eyes, measured along the eye axis and divided by the eye distance
    yaw    = atan(offset / K)

Geometry: the nose tip sits roughly 0.30 x (eye distance) in front of the plane
of the eyes, so for a head turned by theta the offset is 0.30 * tan(theta).
K is that constant (NOSE_PROTRUSION). People differ a little, which is why the
rule engine compares each student against their *own* baseline angle.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

# MediaPipe Face Mesh landmark indices
NOSE_TIP = 1
EYE_OUTER_A = 33     # outer corner of one eye
EYE_OUTER_B = 263    # outer corner of the other eye

NOSE_PROTRUSION = 0.30


def yaw_from_points(nose: Sequence[float], eye_a: Sequence[float], eye_b: Sequence[float],
                    k: float = NOSE_PROTRUSION) -> Optional[float]:
    """Return yaw in degrees (positive = nose toward image-right), or None if degenerate."""
    ax, ay = eye_a
    bx, by = eye_b
    if bx < ax:                      # orient the eye axis left -> right in the image
        ax, ay, bx, by = bx, by, ax, ay
    dx, dy = bx - ax, by - ay
    eye_dist = math.hypot(dx, dy)
    if eye_dist < 1e-6:
        return None
    ux, uy = dx / eye_dist, dy / eye_dist
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    offset = ((nose[0] - mx) * ux + (nose[1] - my) * uy) / eye_dist
    return math.degrees(math.atan(offset / k))
