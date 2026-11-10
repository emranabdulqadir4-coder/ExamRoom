"""Optional performance profiles: pick the one that matches the computer you have.

A profile is only a set of *default* settings. Anything you set yourself (``.env``, ``EXAMGUARD_*``
variables or a command-line flag) always wins over the profile.

    laptop  the default: what Phase 1 was built and tested on (any normal laptop/PC, CPU only)
    edge    a small always-on box in the room (Raspberry Pi 5 class): lighter and slower settings
    jetson  an edge box with an NVIDIA GPU (Jetson class): edge-sized frames, phone detection on the GPU
    gpu     a stronger PC or school server with an NVIDIA GPU: bigger frames, the staged face detector, more FPS

HONEST STATUS: only ``laptop`` has been exercised. ``edge``, ``jetson`` and ``gpu`` are reasoned starting
points that were NOT run on a Raspberry Pi, a Jetson or a GPU server. Treat them as buttons to try: run
``scripts/tune.py`` and ``scripts/evaluate.py`` on your own hardware and adjust.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

SETTINGS_FILE = "settings.json"       # inside the data folder; written by the dashboard's profile buttons


@dataclass(frozen=True)
class Profile:
    name: str
    title: str
    for_whom: str
    tested: bool
    settings: Dict[str, object]


PROFILES: Dict[str, Profile] = {
    "laptop": Profile(
        "laptop", "Laptop / PC", "any normal laptop or PC, no GPU needed", True, {}),
    "edge": Profile(
        "edge", "Edge box (Raspberry Pi 5 class)", "a small box in the room; slower, lighter settings", False,
        {"process_fps": 6.0, "max_width": 640, "phone_enabled": False, "hands_enabled": False,
         "face_detector": "mediapipe", "detector_tiles": 1, "device": "cpu",
         "pre_roll_seconds": 3.5, "post_roll_seconds": 1.5}),
    "jetson": Profile(
        "jetson", "Edge box with GPU (Jetson class)", "a small box with an NVIDIA GPU; phone detection on the GPU", False,
        {"process_fps": 8.0, "max_width": 800, "phone_enabled": True, "yolo_imgsz": 480, "hands_enabled": True,
         "face_detector": "auto", "detector_tiles": 1, "device": "cuda"}),
    "gpu": Profile(
        "gpu", "GPU PC / school server", "a stronger machine with an NVIDIA GPU; bigger frames for big rooms", False,
        {"process_fps": 15.0, "max_width": 1280, "phone_enabled": True, "yolo_imgsz": 960, "hands_enabled": True,
         "face_detector": "staged", "detector_tiles": 2, "device": "cuda"}),
}

DEFAULT_PROFILE = "laptop"


def read_saved_profile(data_dir: Path) -> Optional[str]:
    """The profile chosen with the dashboard buttons, if any (None when missing or unreadable)."""
    try:
        data = json.loads((Path(data_dir) / SETTINGS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = data.get("profile") if isinstance(data, dict) else None
    return name if name in PROFILES else None


def save_profile(data_dir: Path, name: str) -> None:
    if name not in PROFILES:
        raise ValueError("unknown profile: " + name)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / SETTINGS_FILE).write_text(json.dumps({"profile": name}), encoding="utf-8")


def describe(name: str) -> Dict[str, object]:
    p = PROFILES.get(name, PROFILES[DEFAULT_PROFILE])
    return {"name": p.name, "title": p.title, "for_whom": p.for_whom, "tested": p.tested,
            "settings": dict(p.settings)}


def catalogue() -> list:
    return [describe(n) for n in PROFILES]
