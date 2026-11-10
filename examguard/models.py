"""Plain data types shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BBox = tuple  # (x, y, w, h) in processing-frame pixels

KIND_LABELS = {
    "head_turn": "Sustained head turn",
    "repeated_glancing": "Repeated side glancing",
    "phone": "Phone detected",
    "looking_down": "Looking down more than usual",
    "repeated_looking_down": "Repeatedly looking down",
    "eye_gaze": "Eyes glancing sideways",
    "repeated_eye_gaze": "Repeated sideways eye glances",
    "talking": "Talking (mouths moving)",
    "passing": "Possible passing of an object",
}


# Why a teacher dismissed an alert (one tap). Stored with the alert so ExamGuard can learn where it is wrong.
DISMISS_REASONS = {
    "clock": "Looked at the clock",
    "stretching": "Stretching / moving",
    "thinking": "Looking up to think",
    "writing": "Normal writing",
    "bad_detection": "Bad detection (not what it says)",
    "other": "Other",
}
# These mean "the rule fired on ordinary behaviour": the threshold may be too sensitive for this room.
NORMAL_BEHAVIOUR_REASONS = ("clock", "stretching", "thinking", "writing")


@dataclass
class FaceObs:
    """One face seen in one frame."""
    bbox: BBox
    yaw: Optional[float] = None   # degrees; positive = nose toward the right of the image
    track_id: int = 0
    pitch: Optional[float] = None       # degrees; positive = head tilted DOWN
    gaze_x: Optional[float] = None      # iris offset in the eye, about -0.5..0.5; positive = toward image-right
    eyes_down: Optional[float] = None   # 0..1, how far the eyes look down (blendshape score)
    mouth: Optional[float] = None       # mouth opening: lip gap / mouth width (about 0 closed, 0.1-0.5 open)

    @property
    def center(self) -> tuple:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


@dataclass
class Detection:
    """An object found by the object detector (e.g. a phone)."""
    label: str
    conf: float
    bbox: BBox

    @property
    def center(self) -> tuple:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


@dataclass
class Event:
    """A *potential* cheating behaviour. It is only ever a flag for a teacher to review."""
    ts: float
    track_id: int
    kind: str
    severity: str          # "low" | "medium" | "high"
    score: float           # 0..1, used to sort the review queue
    details: str
    bbox: Optional[BBox] = None
    # Filled in when a roster is loaded. Always a *suggestion* that the teacher verifies.
    student_name: Optional[str] = None
    student_ref: Optional[str] = None
    match_score: Optional[float] = None
    # Phase 2: where the student sits and which camera saw it (both optional, both only suggestions)
    seat: Optional[str] = None            # seat label from the seat map, e.g. "B4"
    seat_student: Optional[str] = None    # name of the roster student assigned to that seat
    id_source: Optional[str] = None       # "face" or "seat": where the suggested name came from
    camera: str = ""                      # camera name when the room has two cameras, else ""

    @property
    def label(self) -> str:
        return f"Student {self.track_id}"
