#!/usr/bin/env python3
"""How accurate is ExamGuard in YOUR room? Replay a recorded mock exam and compare its alerts with your labels.

    python scripts/evaluate.py exam.mp4 --events-out alerts.csv        # step 1: list what it flags; scrub the video to those times
    python scripts/evaluate.py exam.mp4 --labels labels.csv             # step 2: score it against what really happened
    python scripts/evaluate.py --demo                                   # self-check of this tool on the simulator

labels.csv (header row, times as seconds or m:ss from the video start):
    start,end,kind
    0:42,0:48,head_turn
kind is one of: head_turn, phone, looking_down, eye_gaze, talking, passing

Add --field to also print what the teachers' decisions say about the same rules in the real room (dismissed
alerts, reasons, reported misses), so lab accuracy and field results sit side by side. That is the *field*
report from the dashboard's "Exam review" page: a dismissal is a teacher's judgement, not proof of a false alarm.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examguard import evaluation as ev   # noqa: E402
from examguard.config import Config      # noqa: E402


def print_field_report(cfg: Config) -> None:
    from examguard import learning
    from examguard.store import Store
    db = cfg.data_path / "examguard.db"
    if not db.is_file():
        print(f"(no field data yet: {db} does not exist)\n")
        return
    store = Store(db)
    try:
        print(learning.format_report(learning.build_report(store, room=cfg.room_name)) + "\n")
    finally:
        store.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--labels")
    ap.add_argument("--events-out", help="write every alert (time, kind, details) to this CSV")
    ap.add_argument("--kinds", help="judge only these behaviours, comma-separated")
    ap.add_argument("--tolerance", type=float, default=3.0, help="seconds of slack around a labelled event (default 3)")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--no-phone", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--field", action="store_true", help="also print the field report built from teacher decisions in data/")
    args = ap.parse_args()
    cfg = Config.from_env(phone_enabled=False if args.no_phone else None)
    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None

    if args.demo:
        events, duration, labels = ev.demo_run(cfg)
        print("SELF-CHECK on the simulator (it knows the true events, so this tests the tool, not your room).\n")
    else:
        if not args.video:
            ap.error("give a video file, or use --demo")
        from examguard.vision import load_detectors, load_face_analyzer
        analyzer = load_face_analyzer(cfg)
        detector = load_detectors(cfg)
        print("Analysing the video (this can take a while)...")
        events, duration = ev.replay(cfg, ev.video_steps(args.video, cfg, analyzer, detector, args.max_seconds))
        labels = ev.load_labels(args.labels) if args.labels else None
    if args.field:
        print_field_report(cfg)
    if args.events_out:
        ev.events_to_csv(events, args.events_out)
        print(f"{len(events)} alerts written to {args.events_out}")
    if labels is None:
        print(f"{len(events)} alerts in {duration:.0f} s. Add --labels labels.csv to score them.")
        return 0
    print(ev.format_report(ev.score(events, labels, duration, args.tolerance, kinds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
