"""Is the camera view good enough for the detectors? (Small faces give unreliable readings.)"""
from __future__ import annotations

import statistics
from typing import Dict, Iterable


def face_health(widths: Iterable[float], min_px: float) -> Dict:
    w = [float(x) for x in widths]
    small = sum(1 for x in w if x < min_px)
    advice = ""
    if small:
        advice = (f"{small} of {len(w)} faces are narrower than {min_px:.0f} px, so detection and head-turn "
                  f"readings for them are unreliable. Move the camera closer or zoom in, use a higher-resolution "
                  f"camera, or (if the camera is higher resolution) raise EXAMGUARD_MAX_WIDTH, e.g. to 1280.")
    return {"n": len(w), "median_px": round(statistics.median(w)) if w else None, "small": small, "advice": advice}
