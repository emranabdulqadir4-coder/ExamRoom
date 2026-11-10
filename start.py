#!/usr/bin/env python3
"""One-click launcher: starts ExamGuard and keeps it running.

    python start.py                  # same options as run.py, e.g.  python start.py --https --host 0.0.0.0

It restarts ExamGuard if it crashes or hangs (and when you press "Apply and restart" on the System check page).
Press Ctrl+C once to stop for good. Use start.bat (Windows) or start.sh (Mac/Linux) to also create the
virtual environment and install the requirements the first time.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESTART_CODE = 75                 # run.py exits with this when it wants to be started again
MAX_QUICK_CRASHES = 5             # this many crashes within QUICK_WINDOW seconds: stop and show the problem
QUICK_WINDOW = 120.0


def check_setup() -> bool:
    missing = []
    for mod in ("cv2", "mediapipe", "flask", "cryptography", "numpy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("Some required packages are missing: " + ", ".join(missing) +
              "\nRun:  pip install -r requirements.txt   (or start with start.bat / start.sh, which does it for you)")
        return False
    return True


def ensure_env_file() -> None:
    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists() and example.exists():
        shutil.copyfile(example, env)
        print("Created .env from .env.example (edit it to add your roster, Gmail alerts and camera address).")


def main(argv: list) -> int:
    if sys.version_info < (3, 10):
        print("ExamGuard needs Python 3.10 or newer.")
        return 1
    if "--demo" not in argv and not check_setup():
        return 1
    ensure_env_file()
    env = dict(os.environ, EXAMGUARD_SUPERVISED="1")
    crashes: list = []
    got_term = {"yes": False}

    def on_term(*_):                       # a service manager asks us to stop: pass it on to ExamGuard
        got_term["yes"] = True
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, on_term)
    except (ValueError, OSError):
        pass
    while True:
        started = time.time()
        proc = subprocess.Popen([sys.executable, str(ROOT / "run.py"), *argv], cwd=str(ROOT), env=env)
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            if got_term["yes"]:
                proc.terminate()           # Ctrl+C already reached the child by itself; SIGTERM did not
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
            print("\nStopped.")
            return 0
        if code == 0 or code in (130, -2, -15):
            return 0                                              # a normal stop
        if code == RESTART_CODE:
            print("\nRestarting ExamGuard ...\n")
            time.sleep(1.0)
            continue
        now = time.time()
        crashes = [t for t in crashes if now - t < QUICK_WINDOW] + [now]
        if len(crashes) >= MAX_QUICK_CRASHES:
            print(f"\nExamGuard stopped {len(crashes)} times in {QUICK_WINDOW:.0f} seconds (last exit code {code}). "
                  f"Not restarting again: read the error above.")
            return code or 1
        wait = min(30, 3 * len(crashes))
        print(f"\nExamGuard stopped unexpectedly (exit code {code}, ran {now - started:.0f} s). Restarting in {wait} s ...\n")
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
