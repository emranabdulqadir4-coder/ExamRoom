"""Video sources: laptop webcam, phone camera over Wi-Fi (DroidCam/EpocCam URL), IP camera, video file.

Live sources capture on a background thread and always hand out the *newest*
frame, so a slow analysis step never builds up latency.
"""
from __future__ import annotations

import logging
import re
import sys
import threading
import time
from typing import Optional, Tuple

import cv2

log = logging.getLogger("examguard.source")


def describe_source(spec: str) -> str:
    """Human-readable source name with any credentials removed."""
    if spec.isdigit():
        return f"camera {spec}"
    return re.sub(r"//[^/@]+@", "//***@", spec)


class LiveSource:
    paced = False

    def __init__(self, spec: str, width: int = 1280, height: int = 720):
        self.spec, self.width, self.height = spec, width, height
        self.state = "starting"
        self._cond = threading.Condition()
        self._frame = None
        self._fid = 0
        self._served = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()

    def _open(self):
        if self.spec.isdigit():
            idx = int(self.spec)
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            cap = cv2.VideoCapture(self.spec)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _loop(self) -> None:
        cap = None
        while not self._stop.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.state = "reconnecting"
                    self._stop.wait(2.0)
                    continue
                self.state = "connected"
                log.info("video source connected: %s", describe_source(self.spec))
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                self.state = "reconnecting"
                self._stop.wait(1.0)
                continue
            with self._cond:
                self._frame, self._fid = frame, self._fid + 1
                self._cond.notify_all()
        if cap is not None:
            cap.release()

    def read(self, timeout: float = 0.5) -> Tuple[Optional[object], int]:
        with self._cond:
            if self._fid == self._served:
                self._cond.wait(timeout)
            if self._fid == self._served:
                return None, -1
            self._served = self._fid
            return self._frame, self._fid

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


class FileSource:
    """A recorded video, played back in real time (handy for testing without a camera)."""
    paced = True

    def __init__(self, path: str, loop: bool = True):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video file: {path}")
        self.loop = loop
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.state = "playing"
        self._fid = 0
        self._due = time.time()

    def read(self, timeout: float = 0.5):
        wait = self._due - time.time()
        if wait > 0:
            time.sleep(min(wait, timeout))
            if self._due > time.time():
                return None, -1
        step = 1.0 / self.fps
        while time.time() > self._due + step and self.cap.grab():     # behind schedule: drop frames
            self._due += step
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            self.state = "ended"
            return None, -1
        self._due += step
        if time.time() - self._due > 1.0:
            self._due = time.time()
        self._fid += 1
        return frame, self._fid

    def close(self) -> None:
        self.cap.release()


def open_source(spec: str, loop: bool = True):
    import os
    if os.path.isfile(spec):
        return FileSource(spec, loop)
    return LiveSource(spec)
