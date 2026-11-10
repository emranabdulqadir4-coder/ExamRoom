#!/usr/bin/env python3
"""Measure ExamGuard on YOURSELF, alone, in a few minutes. No classmates, no hand-written labels.

    python scripts/selftest.py                     # laptop webcam, 3 rounds, about 6 minutes
    python scripts/selftest.py --repeats 5 --phone --talking
    python scripts/selftest.py --source http://192.168.1.50:4747/video
    python scripts/selftest.py --no-score          # only record; score later with scripts/evaluate.py

It shows instructions in a window ("turn your head now"), records a clean video of you, writes the matching
labels file, then runs the same accuracy report as scripts/evaluate.py.
Read the limits in examguard/selftest.py: one person, one seat, one camera position.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examguard import selftest as st          # noqa: E402
from examguard.sources import open_source     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.environ.get("EXAMGUARD_SOURCE", "0"))
    ap.add_argument("--out", default="selftest", help="output name: writes NAME.mp4 and NAME.labels.csv")
    ap.add_argument("--repeats", type=int, default=3, help="rounds of every behaviour (20 labelled events per behaviour need about 20)")
    ap.add_argument("--phone", action="store_true", help="include 'hold up a phone' (needs phone detection installed)")
    ap.add_argument("--talking", action="store_true", help="include talking (scored with talk_min_students=1)")
    ap.add_argument("--no-window", action="store_true", help="do not open a preview window (prompts are printed instead)")
    ap.add_argument("--no-score", action="store_true", help="record only")
    args = ap.parse_args()

    steps = st.build_script(args.repeats, args.phone, args.talking)
    slots, total = st.build_timeline(steps)
    video, labels = f"{args.out}.mp4", f"{args.out}.labels.csv"
    print(f"This takes about {total / 60:.0f} minutes. Sit where a student would sit, facing the camera, and follow the "
          f"instructions on screen.\nPress Q in the window to stop early.")
    input("Press Enter to start (recording begins immediately)... ")

    source = open_source(args.source, loop=False)
    cv2 = None
    show = None
    if not args.no_window:
        try:
            import cv2
            cv2.namedWindow("ExamGuard self-test", cv2.WINDOW_NORMAL)
        except Exception:
            print("(No preview window available: prompts will be printed here.)")
            cv2 = None
    last = {"i": None}

    def show(frame, t):
        info = st.prompt_at(slots, total, t)
        key = (info["phase"], info["i"])
        if cv2 is not None:
            cv2.imshow("ExamGuard self-test", st.draw_prompt(frame, info))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                return False
        elif key != last["i"]:
            print(f"[{t:5.0f}s] {info['title']}: {info['text']}")
        last["i"] = key
        return True

    try:
        length = st.record_session(source, video, slots, total, show=show)
    finally:
        source.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    n = st.write_labels([s for s in slots if s.end <= length], labels)
    print(f"\nRecorded {length:.0f} s to {video} with {n} labelled actions in {labels}.")
    if args.no_score:
        print(f"Score it with:\n  python scripts/evaluate.py {video} --labels {labels}")
        return 0
    env = dict(os.environ)
    if args.talking:
        env["EXAMGUARD_TALK_MIN_STUDENTS"] = "1"
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), video, "--labels", labels,
           "--kinds", ",".join(sorted({k for _, _, k in st.labels_for(slots)}))]
    if not args.phone:
        cmd.append("--no-phone")
    print("\nScoring (this replays the video through the real detectors; it can take a few minutes)...\n")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
