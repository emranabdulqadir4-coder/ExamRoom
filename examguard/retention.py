"""Automatic deletion of evidence once its retention period is over."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .config import Config
from .store import Store
from .vault import Vault

log = logging.getLogger("examguard.retention")


def apply_retention(cfg: Config, store: Store, vault: Vault, now: Optional[float] = None) -> int:
    """Delete due evidence files; keep the (video-free) incident record for the audit trail."""
    n = 0
    for row in store.purge_candidates(cfg, now):
        for name in (row["clip_name"], row["thumb_name"]):
            if name:
                vault.delete(name)
        store.mark_purged(row["id"], expire=(row["status"] == "pending"))
        n += 1
    if n:
        log.info("retention: purged evidence for %d incident(s)", n)
    return n


class RetentionWorker(threading.Thread):
    def __init__(self, cfg: Config, store: Store, vault: Vault, every_seconds: float = 60.0):
        super().__init__(name="retention", daemon=True)
        self.cfg, self.store, self.vault, self.every = cfg, store, vault, every_seconds
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                apply_retention(self.cfg, self.store, self.vault)
            except Exception:                       # never let cleanup crash the app
                log.exception("retention run failed")
            self._stop_evt.wait(self.every)

    def stop(self) -> None:
        self._stop_evt.set()
