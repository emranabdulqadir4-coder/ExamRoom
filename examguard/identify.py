"""Suggest which roster student each tracked face belongs to.

This only ever produces a *suggestion*. The name is shown to the teacher next to the
roster photo, the teacher checks it against the video, and can correct it before
confirming anything. Matching uses OpenCV's SFace embeddings (cosine similarity).

Safeguards against wrong names:
  * only frontal-ish faces (|yaw| < 25 deg) and not-too-small faces are used
  * several samples per student are averaged before a name is suggested
  * the best match must beat a threshold AND beat the runner-up by a margin
  * one roster student can be suggested for only one face at a time
"""
from __future__ import annotations

import logging
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from .config import Config
from .models import FaceObs
from .roster import Roster, Student

log = logging.getLogger("examguard.identify")

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    "face_detection_yunet_2023mar.onnx": BASE + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": BASE + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


@dataclass
class Identity:
    ref: str
    name: str
    student_id: str
    score: float


YUNET = "face_detection_yunet_2023mar.onnx"


def ensure_models(models_dir: Path, only=None) -> Dict[str, Path]:
    """Download the OpenCV face models (or just those named in `only`) on first use."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for fname, url in MODELS.items():
        if only is not None and fname not in only:
            continue
        path = models_dir / fname
        if not (path.exists() and path.stat().st_size > 100_000):
            log.info("downloading %s (one time) ...", fname)
            tmp = path.with_suffix(".part")
            try:
                with urllib.request.urlopen(url, timeout=90) as r, open(tmp, "wb") as f:
                    f.write(r.read())
                if tmp.stat().st_size < 100_000:
                    raise OSError("download too small")
                tmp.replace(path)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not download {fname} ({exc}).\nDownload it from:\n  {url}\n"
                    f"and save it as:\n  {path}") from exc
        out[fname] = path
    return out


class SFaceEmbedder:
    """Face image -> 128-d unit vector (or None if no face is found)."""

    def __init__(self, models_dir: Path):
        paths = ensure_models(models_dir)
        self._det = cv2.FaceDetectorYN.create(str(paths["face_detection_yunet_2023mar.onnx"]), "", (320, 320), 0.6)
        self._rec = cv2.FaceRecognizerSF.create(str(paths["face_recognition_sface_2021dec.onnx"]), "")

    def embed(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = img_bgr.shape[:2]
        if min(h, w) < 160:                               # the detector likes bigger faces
            s = 160.0 / min(h, w)
            img_bgr = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            h, w = img_bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        feat = self._rec.feature(self._rec.alignCrop(img_bgr, face)).flatten().astype(np.float32)
        n = float(np.linalg.norm(feat))
        return feat / n if n > 0 else None


class Identifier:
    MIN_FACE_PX = 40
    MAX_YAW = 25.0
    SAMPLE_EVERY = 0.7          # seconds between samples for one face while still unsure
    REFRESH_EVERY = 12.0        # seconds between samples once a name is suggested
    MIN_SAMPLES = 3
    MAX_PER_FRAME = 2           # bounds CPU use when many faces are visible

    def __init__(self, cfg: Config, roster: Roster, embedder):
        self.cfg, self.roster, self.embedder = cfg, roster, embedder
        self._samples: Dict[int, Deque[np.ndarray]] = {}
        self._last: Dict[int, float] = {}
        self._ident: Dict[int, Optional[Identity]] = {}
        self._enrolled: List[Student] = []

    def for_another_camera(self) -> "Identifier":
        """A second matcher for another camera: same roster and photo embeddings, its own per-face state."""
        other = Identifier(self.cfg, self.roster, self.embedder)
        other._enrolled = list(self._enrolled)
        return other

    # --------------------------------------------------------------- enrol
    def enroll(self) -> List[str]:
        """Compute an embedding for every roster photo. Returns warnings."""
        warnings = list(self.roster.warnings)
        for s in self.roster.students:
            if s.photo_bgr is None:
                continue
            emb = self.embedder.embed(s.photo_bgr)
            if emb is None:
                warnings.append(f"{s.name}: no face found in the photo (face matching skipped)")
                continue
            s.embedding = emb
            self._enrolled.append(s)
        return warnings

    # -------------------------------------------------------------- observe
    def observe(self, ts: float, frame, faces: List[FaceObs]) -> None:
        if not self._enrolled:
            return
        done = 0
        for f in faces:
            if done >= self.MAX_PER_FRAME:
                break
            if f.yaw is None or abs(f.yaw) > self.MAX_YAW or f.bbox[2] < self.MIN_FACE_PX:
                continue
            tid = f.track_id
            wait = self.REFRESH_EVERY if self._ident.get(tid) else self.SAMPLE_EVERY
            if ts - self._last.get(tid, -1e9) < wait:
                continue
            self._last[tid] = ts
            crop = _crop(frame, f.bbox)
            if crop is None:
                continue
            emb = self.embedder.embed(crop)
            done += 1
            if emb is None:
                continue
            self._samples.setdefault(tid, deque(maxlen=8)).append(emb)
            self._ident[tid] = self._decide(tid)
        self._resolve_conflicts()

    def _decide(self, tid: int) -> Optional[Identity]:
        samples = self._samples.get(tid)
        if not samples or len(samples) < self.MIN_SAMPLES:
            return None
        mean = np.mean(samples, axis=0)
        n = float(np.linalg.norm(mean))
        if n == 0:
            return None
        mean /= n
        scored = sorted(((float(np.dot(mean, s.embedding)), s) for s in self._enrolled),
                        key=lambda x: x[0], reverse=True)
        best, student = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0
        if best < self.cfg.match_threshold or best - second < self.cfg.match_margin:
            return None
        return Identity(student.ref, student.name, student.student_id, round(best, 3))

    def _resolve_conflicts(self) -> None:
        """If two faces claim the same student, only the better match keeps the name."""
        claims: Dict[str, int] = {}
        for tid, ident in self._ident.items():
            if ident is None:
                continue
            other = claims.get(ident.ref)
            if other is None:
                claims[ident.ref] = tid
            elif self._ident[other].score >= ident.score:
                self._ident[tid] = None
            else:
                self._ident[other] = None
                claims[ident.ref] = tid

    def identity(self, track_id: int) -> Optional[Identity]:
        return self._ident.get(track_id)

    def names(self) -> Dict[int, str]:
        return {t: i.name for t, i in self._ident.items() if i}


def _crop(frame, bbox, pad: float = 0.35):
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    x0, y0 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
    x1, y1 = min(W, int(x + w * (1 + pad))), min(H, int(y + h * (1 + pad)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return frame[y0:y1, x0:x1].copy()


class FixedIdentifier:
    """Used by the demo: the simulated students have fixed names (no real recognition)."""

    def __init__(self, mapping: Dict[int, Identity]):
        self._map = mapping

    def observe(self, ts, frame, faces) -> None:
        pass

    def identity(self, track_id: int) -> Optional[Identity]:
        return self._map.get(track_id)

    def names(self) -> Dict[int, str]:
        return {t: i.name for t, i in self._map.items()}


def check_roster(roster: Roster, embedder, same_person: float = 0.363, lookalike: float = 0.30) -> List[str]:
    """Problems with the roster photos that would hurt face matching. Empty list = looks fine."""
    issues: List[str] = []
    embs = []
    for s in roster.students:
        if s.photo_bgr is None:
            issues.append(f"{s.name}: no usable photo, so this student will never be suggested by face matching")
            continue
        e = embedder.embed(s.photo_bgr)
        if e is None:
            issues.append(f"{s.name}: no face found in the photo (use a clear, front-facing photo)")
            continue
        embs.append((s, e))
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sim = float(np.dot(embs[i][1], embs[j][1]))
            a, b = embs[i][0].name, embs[j][0].name
            if sim >= same_person:
                issues.append(f"{a} and {b}: the photos look like the SAME person (similarity {sim:.2f}). "
                              f"Wrong or duplicated photo?")
            elif sim >= lookalike:
                issues.append(f"{a} and {b}: the photos look alike (similarity {sim:.2f}); face matching may mix "
                              f"them up, so check these two carefully")
    return issues
