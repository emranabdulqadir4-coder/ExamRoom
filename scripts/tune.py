#!/usr/bin/env python3
"""Measure normal head movement and face size on your camera, then suggest the alert threshold.

    python scripts/tune.py                       # webcam, 60 seconds
    python scripts/tune.py --source exam.mp4 --seconds 120
    python scripts/tune.py --demo                # try it without a camera
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examguard.config import Config   # noqa: E402
from examguard.tuning import pose_check, run   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--pose-check", action="store_true",
                    help="20-second check that head 'down' is read as down (sets EXAMGUARD_PITCH_SIGN)")
    args = ap.parse_args()
    cfg = Config.from_env(source=args.source)
    if args.demo:
        from examguard.demo import DemoWorld
        source = analyzer = DemoWorld(cfg, auto=False)
    else:
        from examguard.sources import open_source
        from examguard.vision import load_face_analyzer
        analyzer = load_face_analyzer(cfg)
        source = open_source(cfg.source, loop=False)
    if args.pose_check:
        pose_check(source, analyzer, cfg)
        source.close()
        return 0
    print(f"Measuring for {args.seconds:.0f} seconds. Ask everyone to sit and write normally...")
    col = run(source, analyzer, cfg.max_width, args.seconds)
    source.close()
    print()
    print(col.report(cfg.yaw_threshold_deg, cfg.min_face_px, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
