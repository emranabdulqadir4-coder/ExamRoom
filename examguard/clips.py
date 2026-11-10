"""Evidence clips: a short, timestamped video around each flagged moment.

A rolling buffer keeps the last few seconds of (already annotated) frames, so a
clip contains the lead-up to the alert plus a short tail. Encoding happens in a
background thread and entirely in memory: the plain video is piped out of ffmpeg
and goes straight into the encrypted vault, so unencrypted footage never touches disk.
"""
from __future__ import annotations

import logging
import queue
import secrets
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .models import Event
from .vault import Vault

log = logging.getLogger("examguard.clips")

OnDone = Callable[[Event, Optional[str], Optional[str]], None]


@dataclass
class _Pending:
    event: Event
    frames: List[Tuple[float, bytes]]
    end_ts: float
    done: bool = field(default=False)


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_mp4(frames_bgr: List[np.ndarray], fps: float) -> bytes:
    """H.264 MP4 (plays in every browser), encoded in memory."""
    h, w = frames_bgr[0].shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)                     # H.264 needs even dimensions
    raw = b"".join(np.ascontiguousarray(f[:h2, :w2]).tobytes() for f in frames_bgr)
    cmd = [_ffmpeg_exe(), "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w2}x{h2}", "-r", f"{fps:.3f}", "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p",
           "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", "pipe:1"]
    proc = subprocess.run(cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg failed: " + proc.stderr.decode(errors="replace")[:300])
    return proc.stdout


def make_thumb(jpeg: bytes, bbox) -> bytes:
    """Small JPEG centred on the flagged student (falls back to the whole frame)."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bbox is not None:
        x, y, w, h = bbox
        pad = 0.9
        x0, y0 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
        x1, y1 = min(img.shape[1], int(x + w * (1 + pad))), min(img.shape[0], int(y + h * (1 + pad)))
        if x1 - x0 > 20 and y1 - y0 > 20:
            img = img[y0:y1, x0:x1]
    scale = 320.0 / max(img.shape[1], 1)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes()


class ClipRecorder:
    def __init__(self, cfg: Config, vault: Vault, on_done: OnDone):
        self.cfg, self.vault, self.on_done = cfg, vault, on_done
        self._ring: Deque[Tuple[float, bytes]] = deque()
        self._pending: List[_Pending] = []
        self._lock = threading.Lock()
        self._jobs: "queue.Queue[Optional[_Pending]]" = queue.Queue()
        self._worker = threading.Thread(target=self._work, name="clip-encoder", daemon=True)
        self._worker.start()

    # --------------------------------------------------------------- feeding
    def push_frame(self, ts: float, jpeg: bytes) -> None:
        ready: List[_Pending] = []
        with self._lock:
            self._ring.append((ts, jpeg))
            while self._ring and ts - self._ring[0][0] > self.cfg.pre_roll_seconds + 0.5:
                self._ring.popleft()
            for p in self._pending:
                p.frames.append((ts, jpeg))
                if ts >= p.end_ts:
                    ready.append(p)
            self._pending = [p for p in self._pending if p not in ready]
        for p in ready:
            self._jobs.put(p)

    def trigger(self, event: Event) -> None:
        """Start a clip for this event, seeded with the pre-roll already in the buffer."""
        with self._lock:
            frames = [f for f in self._ring if f[0] >= event.ts - self.cfg.pre_roll_seconds]
            self._pending.append(_Pending(event, frames, event.ts + self.cfg.post_roll_seconds))

    def tick(self, now: float) -> None:
        """Finish clips whose tail never arrived (e.g. the camera stopped)."""
        ready: List[_Pending] = []
        with self._lock:
            for p in self._pending:
                if now >= p.end_ts + 2.0:
                    ready.append(p)
            self._pending = [p for p in self._pending if p not in ready]
        for p in ready:
            self._jobs.put(p)

    def close(self, flush: bool = True) -> None:
        if flush:
            with self._lock:
                ready, self._pending = self._pending, []
            for p in ready:
                self._jobs.put(p)
        self._jobs.put(None)
        self._worker.join(timeout=60)

    # --------------------------------------------------------------- worker
    def _work(self) -> None:
        while True:
            p = self._jobs.get()
            if p is None:
                return
            clip_name = thumb_name = None
            try:
                clip_name, thumb_name = self._store(p)
            except Exception:
                log.exception("could not build evidence clip")
            try:
                self.on_done(p.event, clip_name, thumb_name)
            except Exception:
                log.exception("incident callback failed")

    def _store(self, p: _Pending) -> Tuple[Optional[str], Optional[str]]:
        frames = p.frames
        if not frames:
            return None, None
        tag = f"{int(p.event.ts * 1000)}-s{p.event.track_id}-{secrets.token_hex(3)}"
        thumb_name = clip_name = None
        ref = min(frames, key=lambda f: abs(f[0] - p.event.ts))
        try:
            thumb_name = self.vault.put(f"thumb-{tag}.jpg", make_thumb(ref[1], p.event.bbox))
        except Exception:
            log.exception("thumbnail failed")
        imgs = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for _, j in frames]
        if len(imgs) == 1:
            imgs = imgs * 2
        span = frames[-1][0] - frames[0][0]
        fps = (len(imgs) - 1) / span if span > 0 and len(frames) > 1 else 10.0
        fps = max(2.0, min(30.0, fps))
        clip_name = self.vault.put(f"clip-{tag}.mp4", encode_mp4(imgs, fps))
        return clip_name, thumb_name
