"""Central configuration.

Every field can be overridden with an ``EXAMGUARD_<NAME>`` environment variable
(or a ``.env`` file in the project root), e.g. ``EXAMGUARD_YAW_THRESHOLD_DEG=40``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Tiny .env reader (no extra dependency). Real environment variables win."""
    path = path or ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Config:
    # --- performance profile (Phase 2) -----------------------------------
    profile: str = ""              # laptop | edge | jetson | gpu ("" = saved dashboard choice, else laptop); see profiles.py
    device: str = "auto"           # where YOLO runs: auto | cpu | cuda

    # --- video source ---------------------------------------------------
    room_name: str = "Exam Room 1"
    source: str = "0"              # webcam index, http(s)/rtsp URL, or video file path
    second_source: str = ""        # OPTIONAL second camera for rooms over ~100 students (other corner); empty = one camera
    second_camera_name: str = "Back camera"
    first_camera_name: str = "Front camera"
    loop_video: bool = True        # loop when the source is a video file
    process_fps: float = 12.0      # analysis rate (frames per second)
    max_width: int = 960           # frames are downscaled to this width for analysis
    max_students: int = 100       # most faces read in one camera view (use a second camera beyond this)

    # --- head-turn rules ------------------------------------------------
    yaw_threshold_deg: float = 35.0   # turn away from the student's own baseline
    turn_seconds: float = 2.0         # sustained turn before an alert
    turn_gap_seconds: float = 1.5     # face may vanish this long mid-turn (turned far away)
    glance_min_seconds: float = 0.6   # shortest turn that counts as a "glance"
    glance_count: int = 4             # glances within the window => "repeated glancing"
    glance_window_seconds: float = 60.0
    use_baseline: bool = True         # compare to each student's usual head angle

    # --- phone rule -----------------------------------------------------
    phone_enabled: bool = True
    phone_confidence: float = 0.40
    phone_seconds: float = 1.0
    yolo_model: str = "yolov8n.pt"
    yolo_imgsz: int = 640            # raise to 960 to catch smaller/farther phones (slower)

    # --- looking down (hidden sheet) -------------------------------------
    look_down_enabled: bool = True
    pitch_sign: float = 1.0           # set to -1 if "looking down" fires when students look UP (see scripts/tune.py --pose-check)
    down_threshold_deg: float = 20.0  # head tilted down this much MORE than the student's usual posture
    eyes_down_threshold: float = 0.35 # ...or eyes looking down this much more than usual (0..1 score)
    down_seconds: float = 3.0
    down_min_seconds: float = 0.6     # shortest dip that counts for "repeated"
    down_count: int = 4
    down_window_seconds: float = 60.0

    # --- eye gaze (eyes sideways while the head faces forward) ----------
    eye_gaze_enabled: bool = True
    gaze_threshold: float = 0.12      # iris shift (fraction of eye width) away from the student's usual
    gaze_seconds: float = 1.5
    gaze_min_seconds: float = 0.4
    gaze_count: int = 5
    gaze_window_seconds: float = 60.0

    # --- talking (several mouths moving at once) ------------------------
    talking_enabled: bool = True
    mouth_amplitude: float = 0.04     # mouth must open/close by at least this much...
    mouth_cycles: int = 3             # ...at least this many times within 2.5 s to count as "moving"
    talk_seconds: float = 3.0
    talk_min_students: int = 2        # students talking at the same time (1 = also flag a single student)
    talk_neighbour_faces: float = 4.0 # "near each other" = within this many face widths

    # --- passing notes / objects (a hand crossing between students) -----
    hands_enabled: bool = True        # needs the hand model (downloads once, about 8 MB)
    pass_hold_seconds: float = 0.4    # the hand must stay near each student at least this long
    pass_window_seconds: float = 3.0

    # --- alerts ---------------------------------------------------------
    cooldown_seconds: float = 20.0    # same student + same rule: at most one alert per window

    # --- evidence clips (3-5 s total, per the proposal) -----------------
    pre_roll_seconds: float = 3.5
    post_roll_seconds: float = 1.5

    # --- retention (privacy) --------------------------------------------
    retention_dismissed_hours: float = 1.0    # dismissed alerts: video deleted after this grace period
    retention_confirmed_days: float = 30.0    # confirmed incidents: video kept for appeals
    retention_pending_days: float = 14.0      # never reviewed: video deleted, alert expires

    # --- storage / web --------------------------------------------------
    data_dir: str = "data"
    host: str = "127.0.0.1"           # use 0.0.0.0 to let other devices on the LAN open the dashboard
    port: int = 5000
    password: str = ""
    public_url: str = ""
    https: bool = False                # serve the dashboard over HTTPS (self-signed certificate unless you give one)
    tls_cert: str = ""
    tls_key: str = ""
    face_detector: str = "auto"        # auto | mediapipe | staged: how faces are found (see README, "Big rooms")
    detector_tiles: int = 1            # staged only: split the picture into n x n tiles so small faces are not lost
    detector_score: float = 0.5        # staged only: how sure the face finder must be (lower finds more, adds false faces)
    crop_min_side: int = 192           # staged only: each face is enlarged to at least this many pixels before reading it
    min_face_px: int = 40              # faces narrower than this are flagged as unreliable on the dashboard

    # --- Student roster + face matching (optional) -----------------------
    roster_file: str = ""              # .xlsx or .csv with name, student_id, photo (see roster.py)
    roster_photos_dir: str = ""        # default: "photos" folder next to the roster file
    match_threshold: float = 0.363     # cosine similarity needed to suggest a name (OpenCV SFace default)
    match_margin: float = 0.05         # best match must beat the runner-up by this much

    write_marks: bool = True           # when a teacher confirms, write the mark into the roster file
    marks_column: str = "mark"         # column header to write into (created if missing)
    mark_value: str = "Fail"

    # --- Seat map (optional) ---------------------------------------------
    seat_radius_faces: float = 1.6     # a face belongs to a seat when it is within this many face widths of the seat mark

    # --- Edge box <-> server split over MQTT (Phase 2, optional; see README "Edge box + server") ----------
    role: str = "standalone"           # standalone (one computer, the default) | edge (camera + AI) | server (dashboard)
    mqtt_host: str = ""                # the broker's address, for edge and server
    mqtt_port: int = 1883
    mqtt_user: str = ""
    mqtt_password: str = ""
    mqtt_tls: bool = False
    mqtt_cafile: str = ""              # CA certificate for a broker with a private certificate
    mqtt_topic_prefix: str = "examguard"
    mqtt_key: str = ""                 # shared secret that encrypts everything sent through the broker (make_key.py)
    node_name: str = ""                # a name for this edge box (default: the computer's name)
    outbox_max_mb: int = 1024          # edge: undelivered alerts kept on disk while the server is unreachable
    heartbeat_seconds: float = 5.0     # edge: how often it reports its status
    live_seconds: float = 2.0          # edge: how often it sends a live picture (the server's live view is a slow slideshow)

    # --- Learning from teacher decisions (Phase 2) -----------------------
    admin_names: str = ""              # comma-separated names allowed to approve suggested threshold changes (empty = any teacher)

    # --- Watchdog (used by start.py) -------------------------------------
    watchdog_seconds: float = 90.0     # camera connected but no new frames for this long -> restart ExamGuard

    # --- Email alerts via Gmail (optional) ------------------------------
    email_user: str = ""               # the Gmail address that sends the alerts
    email_app_password: str = ""       # a Google "App password" (NOT your normal password)
    email_to: str = ""                 # who receives alerts; comma-separated for several teachers
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 465         # SSL
    email_send_thumb: bool = False     # off by default: do not email student images

    def __post_init__(self) -> None:
        if not self.profile:                 # built directly (tests, scripts) instead of through from_env
            self.profile = "laptop"

    # ------------------------------------------------------------------
    @property
    def scheme(self) -> str:
        return "https" if self.https else "http"

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else ROOT / p

    @property
    def models_path(self) -> Path:
        return ROOT / "models"

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        from .profiles import DEFAULT_PROFILE, PROFILES, read_saved_profile
        load_dotenv()
        kwargs: dict = {}
        for f in fields(cls):
            raw = os.environ.get("EXAMGUARD_" + f.name.upper())
            if raw is None:
                continue
            default = f.default
            if isinstance(default, bool):
                kwargs[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                kwargs[f.name] = int(raw)
            elif isinstance(default, float):
                kwargs[f.name] = float(raw)
            else:
                kwargs[f.name] = raw
        kwargs.update({k: v for k, v in overrides.items() if v is not None})

        data_dir = kwargs.get("data_dir", "data")
        data_path = Path(data_dir) if Path(data_dir).is_absolute() else ROOT / data_dir
        # Threshold changes a teacher approved on the dashboard: defaults too, anything set explicitly wins.
        from .learning import read_overrides
        for key, value in read_overrides(data_path).items():
            kwargs.setdefault(key, value)

        # Performance profile: its settings are only defaults, anything set explicitly above wins.
        name = (kwargs.get("profile") or "").strip().lower() or read_saved_profile(data_path) or DEFAULT_PROFILE
        if name not in PROFILES:
            raise ValueError(f"EXAMGUARD_PROFILE must be one of {', '.join(PROFILES)} (got '{name}')")
        for key, value in PROFILES[name].settings.items():
            kwargs.setdefault(key, value)
        kwargs["profile"] = name
        return cls(**kwargs)
