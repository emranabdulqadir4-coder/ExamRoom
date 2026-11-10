"""Notices when ExamGuard is alive but no longer analysing video, so the launcher can restart it.

Normal camera drops are handled by the video source itself (it reconnects). This catches the rare hang:
the camera claims to be connected, yet no new picture has arrived for ``watchdog_seconds``, or processing
has been failing the whole time.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional

log = logging.getLogger("examguard.watchdog")


class Watchdog:
    def __init__(self, limit_seconds: float, monitors: List, on_trip: Callable[[str], None],
                 interval: float = 10.0, clock: Callable[[], float] = time.time):
        self.limit, self.monitors, self.on_trip = limit_seconds, monitors, on_trip
        self.interval, self.clock = interval, clock
        self._error_since: Dict[int, Optional[float]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def check(self) -> Optional[str]:
        """The reason to restart, or None when everything looks alive."""
        now = self.clock()
        for i, m in enumerate(self.monitors):
            st = m.status()
            name = st.get("camera") or "camera"
            age = st.get("last_frame_age")
            connected = st.get("source_state") in ("connected", "playing", "simulation")
            if connected and age is not None and age > self.limit:
                return f"{name}: connected but no new picture for {age:.0f} s"
            if connected and age is None and st.get("uptime", 0) > self.limit and st.get("state") != "error":
                return f"{name}: connected but never produced a picture in {st['uptime']} s"
            if st.get("state") == "error":
                since = self._error_since.setdefault(i, now)
                if now - since > self.limit:
                    return f"{name}: processing has been failing for {now - since:.0f} s ({st.get('error')})"
            else:
                self._error_since[i] = None
        return None

    def start(self) -> None:
        def run():
            while not self._stop.wait(self.interval):
                reason = self.check()
                if reason:
                    log.error("watchdog: %s -> restarting", reason)
                    self.on_trip(reason)
                    return
        self._thread = threading.Thread(target=run, name="watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
