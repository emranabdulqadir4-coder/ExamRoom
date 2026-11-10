"""Push channel from the monitor to every open dashboard (Server-Sent Events)."""
from __future__ import annotations

import json
import queue
import threading
from typing import List


class AlertHub:
    def __init__(self):
        self._subs: List["queue.Queue[tuple]"] = []
        self._lock = threading.Lock()

    def subscribe(self) -> "queue.Queue[tuple]":
        q: "queue.Queue[tuple]" = queue.Queue(maxsize=200)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event_type: str, data: dict) -> None:
        payload = (event_type, json.dumps(data))
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:                      # a stuck client must never block monitoring
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except queue.Empty:
                    pass

    @property
    def clients(self) -> int:
        with self._lock:
            return len(self._subs)
