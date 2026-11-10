"""Real computer vision: MediaPipe face landmarks (head turn) and YOLO (phone).

Both are optional at import time so the rest of the app (and the demo) runs
without them; missing pieces produce clear instructions instead of tracebacks.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import List, Optional

import cv2

from .config import Config
from .facefeatures import eyes_down_score, gaze_ratio, mouth_openness, pitch_from_matrix
from .health import face_health
from .headpose import EYE_OUTER_A, EYE_OUTER_B, NOSE_TIP, yaw_from_points
from .models import Detection, FaceObs

log = logging.getLogger("examguard.vision")

FACE_MODEL_NAME = "face_landmarker.task"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")


def ensure_face_model(models_dir: Path) -> Path:
    """Return the MediaPipe face model, downloading it (about 4 MB) on first use."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / FACE_MODEL_NAME
    if path.exists() and path.stat().st_size > 100_000:
        return path
    log.info("downloading face model (one time) ...")
    tmp = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(FACE_MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the face model ({exc}).\n"
            f"Download it manually from:\n  {FACE_MODEL_URL}\n"
            f"and save it as:\n  {path}") from exc
    return path


def obs_from_points(pts, matrix=None, shapes=None, pitch_sign: float = 1.0) -> FaceObs:
    """Landmarks in frame pixels (478 x,y pairs) -> one FaceObs with every reading the rules use."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return FaceObs(
        bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
        yaw=yaw_from_points(pts[NOSE_TIP], pts[EYE_OUTER_A], pts[EYE_OUTER_B]),
        pitch=pitch_from_matrix(matrix, pitch_sign) if matrix is not None else None,
        gaze_x=gaze_ratio(pts), eyes_down=eyes_down_score(shapes or {}), mouth=mouth_openness(pts))


class FaceAnalyzer:
    """Finds every face in the frame and estimates how far each head is turned."""
    name = "MediaPipe FaceLandmarker"

    def __init__(self, cfg: Config, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=cfg.max_students,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_face_blendshapes=True,                  # eye-look-down scores
            output_facial_transformation_matrixes=True,    # head pitch
        )
        self.pitch_sign = cfg.pitch_sign
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def analyze(self, frame, ts_ms: int) -> List[FaceObs]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        faces: List[FaceObs] = []
        matrices = getattr(result, "facial_transformation_matrixes", None)
        blend = getattr(result, "face_blendshapes", None)
        for i, lms in enumerate(result.face_landmarks):
            pts = [(p.x * w, p.y * h) for p in lms]
            matrix = matrices[i] if matrices is not None and i < len(matrices) else None
            shapes = {c.category_name: c.score for c in blend[i]} if blend is not None and i < len(blend) else {}
            faces.append(obs_from_points(pts, matrix, shapes, self.pitch_sign))
        return faces

    def close(self) -> None:
        self._landmarker.close()


def resolve_device(wanted: str) -> Optional[str]:
    """Where YOLO should run. 'auto' lets Ultralytics choose; 'cuda' falls back to the CPU (with a warning) if no GPU is usable."""
    wanted = (wanted or "auto").strip().lower()
    if wanted == "auto":
        return None
    if wanted == "cpu":
        return "cpu"
    if wanted in ("cuda", "gpu", "0"):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        log.warning("EXAMGUARD_DEVICE=%s was requested but no usable NVIDIA GPU was found: using the CPU instead", wanted)
        return "cpu"
    raise ValueError("EXAMGUARD_DEVICE must be auto, cpu or cuda (got '%s')" % wanted)


class PhoneDetector:
    """YOLO (Ultralytics) restricted to the COCO 'cell phone' class."""
    name = "YOLO"

    def __init__(self, cfg: Config, models_dir: Path):
        from ultralytics import YOLO
        weights = Path(models_dir) / cfg.yolo_model
        try:
            self.model = YOLO(str(weights))          # downloads into models/ the first time
        except Exception:
            log.warning("could not use %s, falling back to the default download location", weights)
            self.model = YOLO(cfg.yolo_model)
        names = self.model.names
        self.class_id = next((i for i, n in names.items() if n == "cell phone"), 67)
        self.conf, self.imgsz = cfg.phone_confidence, cfg.yolo_imgsz
        self.device = resolve_device(cfg.device)

    def detect(self, frame) -> List[Detection]:
        extra = {"device": self.device} if self.device else {}
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 classes=[self.class_id], verbose=False, **extra)[0]
        out: List[Detection] = []
        for box in res.boxes:
            x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
            out.append(Detection("cell phone", float(box.conf[0]), (int(x0), int(y0), int(x1 - x0), int(y1 - y0))))
        return out


HAND_MODEL_NAME = "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")


def ensure_hand_model(models_dir: Path) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / HAND_MODEL_NAME
    if path.exists() and path.stat().st_size > 100_000:
        return path
    log.info("downloading hand model (one time) ...")
    tmp = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(HAND_MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"could not download the hand model ({exc}). Download {HAND_MODEL_URL} "
                           f"and save it as {path}") from exc
    return path


class HandDetector:
    """MediaPipe HandLandmarker: where are the hands? Used to notice objects passed between students."""
    name = "hands"

    def __init__(self, cfg: Config, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        self._last_ms = 0
        self._lm = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=min(2 * cfg.max_students, 24),
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))

    def detect(self, frame) -> List[Detection]:
        import time
        h, w = frame.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._last_ms = max(self._last_ms + 1, int(time.time() * 1000))
        res = self._lm.detect_for_video(image, self._last_ms)
        out: List[Detection] = []
        for i, lms in enumerate(res.hand_landmarks):
            xs, ys = [p.x * w for p in lms], [p.y * h for p in lms]
            conf = res.handedness[i][0].score if i < len(res.handedness) and res.handedness[i] else 0.5
            out.append(Detection("hand", float(conf), (int(min(xs)), int(min(ys)),
                                                       int(max(xs) - min(xs)), int(max(ys) - min(ys)))))
        return out


class MultiDetector:
    """Runs several object detectors; one failing never silences the others."""

    def __init__(self, parts: dict):
        self.parts = parts
        self.names = list(parts)
        self._warned = set()

    def detect(self, frame) -> List[Detection]:
        out: List[Detection] = []
        for name, det in self.parts.items():
            try:
                out += det.detect(frame)
            except Exception as exc:
                if name not in self._warned:
                    self._warned.add(name)
                    log.warning("%s detection failed and will keep retrying: %s", name, exc)
        return out


def load_detectors(cfg: Config) -> Optional[MultiDetector]:
    parts = {}
    phone = load_phone_detector(cfg)
    if phone is not None:
        parts["phone"] = phone
    if cfg.hands_enabled:
        try:
            parts["hands"] = HandDetector(cfg, ensure_hand_model(cfg.models_path))
        except Exception as exc:
            log.warning("Passing-an-object detection is OFF: %s", exc)
    return MultiDetector(parts) if parts else None


def load_phone_detector(cfg: Config) -> Optional[PhoneDetector]:
    if not cfg.phone_enabled:
        return None
    try:
        return PhoneDetector(cfg, cfg.models_path)
    except ImportError:
        log.warning("Phone detection is OFF: 'ultralytics' is not installed. "
                    "Run: pip install -r requirements-phone.txt   (or start with --no-phone)")
    except Exception as exc:
        log.warning("Phone detection is OFF: could not load the YOLO model (%s)", exc)
    return None


# =====================================================================================
# Stronger face finding for big rooms: find faces first (YuNet), then analyse each one
# in its own enlarged crop. Faces that are small in the picture become large for MediaPipe.
# =====================================================================================
def tile_rects(w: int, h: int, n: int, overlap: float = 0.2) -> List[tuple]:
    """Split a frame into an n x n grid of overlapping tiles (x0, y0, x1, y1). n=1 returns the whole frame."""
    if n <= 1:
        return [(0, 0, w, h)]
    tw, th = w / n, h / n
    ox, oy = tw * overlap, th * overlap
    return [(int(max(0, c * tw - ox)), int(max(0, r * th - oy)), int(min(w, (c + 1) * tw + ox)),
             int(min(h, (r + 1) * th + oy))) for r in range(n) for c in range(n)]


def _iou(a, b) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax1, bx1) - max(a[0], b[0]), min(ay1, by1) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def merge_boxes(boxes: List[tuple], iou: float = 0.4) -> List[tuple]:
    """Drop duplicate detections of one face (from overlapping tiles). Boxes are (x, y, w, h, score)."""
    kept: List[tuple] = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(_iou(b, k) < iou for k in kept):
            kept.append(b)
    return kept


def crop_for_face(frame, box, pad: float = 0.4, min_side: int = 192):
    """Square crop around a face, enlarged to at least min_side pixels.

    Returns (crop, x0, y0, scale): a point (px, py) in the crop is (x0 + px/scale, y0 + py/scale) in the frame."""
    fh, fw = frame.shape[:2]
    x, y, w, h = box[:4]
    side = max(w, h) * (1 + 2 * pad)
    side = int(max(16, min(side, fw, fh)))
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = int(min(max(0, cx - side / 2.0), fw - side))
    y0 = int(min(max(0, cy - side / 2.0), fh - side))
    crop = frame[y0:y0 + side, x0:x0 + side]
    scale = max(1.0, min_side / float(side))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, x0, y0, scale


class YuNetDetector:
    """OpenCV's YuNet face detector, optionally run on overlapping tiles so small faces are not lost."""
    name = "YuNet"

    def __init__(self, cfg: Config, model_path: Path):
        self._det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), cfg.detector_score)
        self.tiles = max(1, cfg.detector_tiles)

    def detect(self, frame) -> List[tuple]:
        h, w = frame.shape[:2]
        found: List[tuple] = []
        for (x0, y0, x1, y1) in tile_rects(w, h, self.tiles):
            tile = frame[y0:y1, x0:x1]
            self._det.setInputSize((tile.shape[1], tile.shape[0]))
            _, faces = self._det.detect(tile)
            for f in (faces if faces is not None else []):
                found.append((float(f[0]) + x0, float(f[1]) + y0, float(f[2]), float(f[3]), float(f[-1])))
        return merge_boxes(found) if self.tiles > 1 else found


class CropLandmarker:
    """MediaPipe FaceLandmarker on a single enlarged face crop (no video tracking between calls)."""

    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        self._lm = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE, num_faces=3,
            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))

    def __call__(self, crop):
        """-> (points in crop pixels, matrix, blendshape dict) for the face nearest the crop centre, or None."""
        ch, cw = crop.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = self._lm.detect(image)
        best, best_d = None, 1e18
        for i, lms in enumerate(res.face_landmarks):
            pts = [(p.x * cw, p.y * ch) for p in lms]
            mx = sum(p[0] for p in pts) / len(pts) - cw / 2.0
            my = sum(p[1] for p in pts) / len(pts) - ch / 2.0
            d = mx * mx + my * my
            if d < best_d:
                matrices = getattr(res, "facial_transformation_matrixes", None)
                blend = getattr(res, "face_blendshapes", None)
                shapes = {c.category_name: c.score for c in blend[i]} if blend is not None and i < len(blend) else {}
                best, best_d = (pts, matrices[i] if matrices is not None and i < len(matrices) else None, shapes), d
        return best

    def close(self) -> None:
        self._lm.close()


class StagedFaceAnalyzer:
    """Find faces with a detector, then read each face from its own enlarged crop."""

    def __init__(self, cfg: Config, detector, landmarks):
        self.cfg, self.detector, self.landmarks = cfg, detector, landmarks
        self.pitch_sign = cfg.pitch_sign
        self.name = f"{getattr(detector, 'name', 'detector')} + MediaPipe (per-face crops)"

    def analyze(self, frame, ts_ms: int) -> List[FaceObs]:
        boxes = sorted(self.detector.detect(frame), key=lambda b: -b[4])[: self.cfg.max_students]
        faces: List[FaceObs] = []
        for box in boxes:
            crop, x0, y0, scale = crop_for_face(frame, box, min_side=self.cfg.crop_min_side)
            res = self.landmarks(crop)
            if res is None:
                # Face found but too hard to read: keep it so it is counted (and the small-face warning stays honest).
                faces.append(FaceObs(bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3]))))
                continue
            pts, matrix, shapes = res
            full = [(x0 + px / scale, y0 + py / scale) for px, py in pts]
            faces.append(obs_from_points(full, matrix, shapes, self.pitch_sign))
        faces.sort(key=lambda f: (f.bbox[1], f.bbox[0]))
        return faces

    def close(self) -> None:
        close = getattr(self.landmarks, "close", None)
        if close:
            close()


class AutoFaceAnalyzer:
    """Plain MediaPipe while faces are big enough; switches (once) to the staged analyser if they stay small."""
    SMALL_SHARE = 0.2                      # this share of faces below min_face_px counts as "too small"

    def __init__(self, cfg: Config, primary, make_staged, switch_after_frames: Optional[int] = None):
        self.cfg, self._primary, self._make_staged = cfg, primary, make_staged
        self._active, self._staged = primary, None
        self._bad = 0
        self._need = switch_after_frames or max(20, int(cfg.process_fps * 10))
        self._gave_up = False

    @property
    def name(self) -> str:
        return getattr(self._active, "name", "analyzer") + (" (auto)" if not self._gave_up else "")

    @property
    def switched(self) -> bool:
        return self._active is not self._primary

    def analyze(self, frame, ts_ms: int) -> List[FaceObs]:
        faces = self._active.analyze(frame, ts_ms)
        if self._active is self._primary and not self._gave_up:
            small = face_health([f.bbox[2] for f in faces], self.cfg.min_face_px)["small"]
            self._bad = self._bad + 1 if faces and small / len(faces) >= self.SMALL_SHARE else 0
            if self._bad >= self._need:
                try:
                    self._staged = self._make_staged()
                    self._active = self._staged
                    log.info("faces are small: switched to the staged face detector")
                except Exception as exc:
                    self._gave_up = True
                    log.warning("small faces detected but the stronger detector is unavailable (%s); "
                                "continuing with MediaPipe only", exc)
        return faces

    def close(self) -> None:
        for a in (self._primary, self._staged):
            close = getattr(a, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass


def _make_staged(cfg: Config) -> StagedFaceAnalyzer:
    from .identify import YUNET, ensure_models
    yunet = ensure_models(cfg.models_path, only=(YUNET,))[YUNET]
    return StagedFaceAnalyzer(cfg, YuNetDetector(cfg, yunet), CropLandmarker(ensure_face_model(cfg.models_path)))


def load_face_analyzer(cfg: Config):
    """EXAMGUARD_FACE_DETECTOR: mediapipe (default behaviour), staged (always the stronger detector), or auto."""
    mode = cfg.face_detector.strip().lower()
    if mode not in ("auto", "mediapipe", "staged"):
        raise ValueError(f"EXAMGUARD_FACE_DETECTOR must be auto, mediapipe or staged (got '{cfg.face_detector}')")
    if mode == "staged":
        return _make_staged(cfg)                     # asked for explicitly: fail loudly if it cannot start
    primary = FaceAnalyzer(cfg, ensure_face_model(cfg.models_path))
    if mode == "mediapipe":
        return primary
    return AutoFaceAnalyzer(cfg, primary, lambda: _make_staged(cfg))
