#!/usr/bin/env python3
"""Check the roster photos before an exam: missing photos, photos without a face, duplicate or look-alike photos.

    python scripts/check_roster.py roster/roster.xlsx [--photos roster/photos]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examguard.config import Config                      # noqa: E402
from examguard.identify import SFaceEmbedder, check_roster   # noqa: E402
from examguard.roster import Roster                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roster")
    ap.add_argument("--photos", default="")
    args = ap.parse_args()
    cfg = Config.from_env()
    roster = Roster.load(args.roster, args.photos)
    issues = check_roster(roster, SFaceEmbedder(cfg.models_path), same_person=cfg.match_threshold)
    print(f"{len(roster)} students in the roster.")
    if not issues:
        print("No problems found.")
        return 0
    print(f"{len(issues)} thing(s) to fix or watch:")
    for i in issues:
        print("  -", i)
    return 1


if __name__ == "__main__":
    sys.exit(main())
