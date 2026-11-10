"""Learning from teacher decisions (Phase 2): per-rule report, suggested threshold changes, approved overrides.

Principles (from the project plan):
  * Nothing changes by itself. A suggestion is only a proposal; a teacher / administrator approves or rejects it,
    and an approved change takes effect the next time ExamGuard starts.
  * Only numbers are used here: rule, room, outcome, reason, who decided. No names, no seats, no video.
  * A dismissal is NOT proof that nothing happened (the teacher may have been wrong or hurried), and an alert
    that was only ignored or never reviewed is NOT a false alarm: it is left out of every rate.
  * Teachers differ. The report shows every teacher's own dismissal rate, and no suggestion is made for a rule
    when teachers clearly disagree about it.
"""
from __future__ import annotations

import csv
import io
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .models import DISMISS_REASONS, KIND_LABELS, NORMAL_BEHAVIOUR_REASONS
from .store import ReviewError, Store

OVERRIDES_FILE = "threshold_overrides.json"     # inside the data folder; written only when a suggestion is approved

MIN_DECIDED = 20            # decided alerts for one rule before any suggestion (a handful of alerts proves nothing)
MIN_MISSED = 3              # missed-event reports needed before suggesting to be MORE sensitive
RAISE_IF_NORMAL_SHARE = 0.5     # >= half of the decided alerts were dismissed because it was normal behaviour
LOWER_IF_NORMAL_SHARE = 0.15    # ...and this few, with missed reports, means the rule may be too strict
TEACHER_MIN_DECIDED = 5         # a teacher needs this many decisions to count in the disagreement check
TEACHER_SPREAD = 0.35           # teachers whose rates differ by more than this "disagree"


@dataclass(frozen=True)
class Knob:
    """The one setting that is changed for a rule, and how far one step goes."""
    setting: str
    step: float
    lo: float
    hi: float
    integer: bool = False
    meaning: str = ""


# raise = less sensitive (fewer alerts). One small step at a time, always inside safe limits.
KNOBS: Dict[str, Knob] = {
    "head_turn": Knob("yaw_threshold_deg", 5.0, 20.0, 60.0, False, "degrees the head must turn"),
    "repeated_glancing": Knob("glance_count", 1, 2, 10, True, "glances needed within the window"),
    "phone": Knob("phone_confidence", 0.05, 0.25, 0.80, False, "how sure the phone detector must be"),
    "looking_down": Knob("down_threshold_deg", 3.0, 10.0, 40.0, False, "degrees the head must tilt down"),
    "repeated_looking_down": Knob("down_count", 1, 2, 10, True, "downward looks needed within the window"),
    "eye_gaze": Knob("gaze_threshold", 0.02, 0.06, 0.30, False, "how far the eyes must point sideways"),
    "repeated_eye_gaze": Knob("gaze_count", 1, 2, 12, True, "sideways glances needed within the window"),
    "talking": Knob("talk_seconds", 1.0, 2.0, 10.0, False, "seconds the mouths must keep moving"),
    "passing": Knob("pass_hold_seconds", 0.2, 0.2, 2.0, False, "seconds the hand must stay near each student"),
}
ALLOWED_SETTINGS = {k.setting: k for k in KNOBS.values()}


# ---------------------------------------------------------------- report
def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 3) if den else None


def build_report(store: Store, room: Optional[str] = None, since: Optional[float] = None,
                 exam_id: Optional[int] = None) -> Dict:
    rows = store.learning_rows(room, since, exam_id)
    missed = store.missed_rows(room, since, exam_id)

    groups: Dict[tuple, Dict] = {}

    def grp(r_room: str, kind: str) -> Dict:
        key = (r_room, kind)
        if key not in groups:
            groups[key] = {"room": r_room, "kind": kind,
                           "kind_label": KIND_LABELS.get(kind, "Something else" if kind == "other" else kind),
                           "alerts": 0, "confirmed": 0, "dismissed": 0, "normal_behaviour": 0, "bad_detection": 0,
                           "other_reason": 0, "no_reason": 0, "not_decided": 0, "missed": 0,
                           "reasons": {}}
        return groups[key]

    teachers: Dict[str, Dict] = defaultdict(lambda: {"decided": 0, "confirmed": 0, "dismissed": 0, "normal_behaviour": 0})
    per_teacher_rule: Dict[tuple, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {"decided": 0, "normal": 0}))

    for r in rows:
        g = grp(r["room"], r["kind"])
        g["alerts"] += 1
        st = r["status"]
        if st == "confirmed":
            g["confirmed"] += 1
        elif st == "dismissed":
            g["dismissed"] += 1
            reason = r["dismiss_reason"]
            if reason in NORMAL_BEHAVIOUR_REASONS:
                g["normal_behaviour"] += 1
            elif reason == "bad_detection":
                g["bad_detection"] += 1
            elif reason == "other":
                g["other_reason"] += 1
            else:
                g["no_reason"] += 1
            g["reasons"][reason or "none"] = g["reasons"].get(reason or "none", 0) + 1
        else:                                  # still pending (including ignored) or expired: no verdict
            g["not_decided"] += 1
        if st in ("confirmed", "dismissed") and r["reviewer"]:
            t = teachers[r["reviewer"]]
            t["decided"] += 1
            t[st] += 1
            t["normal_behaviour"] += 1 if st == "dismissed" and r["dismiss_reason"] in NORMAL_BEHAVIOUR_REASONS else 0
            pr = per_teacher_rule[(r["room"], r["kind"])][r["reviewer"]]
            pr["decided"] += 1
            pr["normal"] += 1 if st == "dismissed" and r["dismiss_reason"] in NORMAL_BEHAVIOUR_REASONS else 0
    for m in missed:
        grp(m["room"], m["kind"])["missed"] += 1

    rules = []
    for key in sorted(groups, key=lambda k: (k[0], -groups[k]["alerts"], k[1])):
        g = groups[key]
        decided = g["confirmed"] + g["dismissed"]
        g["decided"] = decided
        g["dismissed_share"] = _rate(g["dismissed"], decided)
        g["normal_share"] = _rate(g["normal_behaviour"], decided)
        g["teacher_spread"] = _spread(per_teacher_rule[key])
        rules.append(g)
    teacher_rows = [{"reviewer": name, **t, "dismissed_share": _rate(t["dismissed"], t["decided"])}
                    for name, t in sorted(teachers.items())]
    totals = {"alerts": len(rows), "decided": sum(g["decided"] for g in rules),
              "confirmed": sum(g["confirmed"] for g in rules), "dismissed": sum(g["dismissed"] for g in rules),
              "not_decided": sum(g["not_decided"] for g in rules), "missed": len(missed)}
    return {
        "scope": {"room": room, "since": since, "exam_id": exam_id, "made": time.time()},
        "rules": rules, "teachers": teacher_rows, "totals": totals,
        "reasons": {k: v for k, v in DISMISS_REASONS.items()},
        "notes": [
            "A dismissal is a teacher's judgement, not proof that nothing happened.",
            "Alerts that are still waiting, ignored, or expired are not counted as false alarms.",
            "Missed-event reports only cover what a teacher noticed; the real number of misses is unknown.",
            "'Bad detection' dismissals are not used to change thresholds (a different threshold would not fix them).",
        ],
    }


def _spread(by_teacher: Dict[str, Dict]) -> Optional[float]:
    rates = [v["normal"] / v["decided"] for v in by_teacher.values() if v["decided"] >= TEACHER_MIN_DECIDED]
    return round(max(rates) - min(rates), 3) if len(rates) >= 2 else None


def report_csv(report: Dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["room", "rule", "alerts", "confirmed", "dismissed", "dismissed_normal_behaviour", "dismissed_bad_detection",
                "dismissed_other", "dismissed_no_reason", "not_decided_or_ignored", "missed_reports",
                "dismissed_share_of_decided", "normal_behaviour_share_of_decided"])
    for g in report["rules"]:
        w.writerow([g["room"], g["kind"], g["alerts"], g["confirmed"], g["dismissed"], g["normal_behaviour"],
                    g["bad_detection"], g["other_reason"], g["no_reason"], g["not_decided"], g["missed"],
                    "" if g["dismissed_share"] is None else g["dismissed_share"],
                    "" if g["normal_share"] is None else g["normal_share"]])
    return "﻿" + buf.getvalue()


def format_report(report: Dict) -> str:
    lines = ["Field results from teacher decisions (not a lab test)", ""]
    if not report["rules"]:
        return "\n".join(lines + ["No alerts or missed reports yet."])
    for g in report["rules"]:
        share = "n/a" if g["dismissed_share"] is None else f"{g['dismissed_share']*100:.0f}%"
        lines.append(f"  {g['room']} / {g['kind_label']}: {g['alerts']} alerts, {g['confirmed']} confirmed, "
                     f"{g['dismissed']} dismissed ({share} of decided; {g['normal_behaviour']} normal behaviour, "
                     f"{g['bad_detection']} bad detection), {g['not_decided']} undecided/ignored, {g['missed']} missed reports")
    lines.append("")
    lines += ["  " + n for n in report["notes"]]
    return "\n".join(lines)


# ----------------------------------------------------------- suggestions
def _round(knob: Knob, v: float) -> float:
    v = int(round(v)) if knob.integer else round(v, 3)
    return float(min(knob.hi, max(knob.lo, v)))


def propose(store: Store, cfg: Config, now: Optional[float] = None) -> List[Dict]:
    """Create pending suggestions for this room from the teachers' decisions. Never changes a setting.

    Only decisions made after the last approved or rejected suggestion for the same setting are used, so a
    suggestion is never repeated on the strength of data that has already been acted on.
    """
    created: List[Dict] = []
    existing = store.list_suggestions(room=cfg.room_name)
    for kind, knob in KNOBS.items():
        if any(s["setting"] == knob.setting and s["status"] == "pending" for s in existing):
            continue
        last_decided = max([s["decided_ts"] for s in existing if s["setting"] == knob.setting and s["decided_ts"]] or [0.0])
        rep = build_report(store, room=cfg.room_name, since=last_decided or None)
        g = next((x for x in rep["rules"] if x["kind"] == kind and x["room"] == cfg.room_name), None)
        if g is None or g["decided"] < MIN_DECIDED:
            continue
        if g["teacher_spread"] is not None and g["teacher_spread"] > TEACHER_SPREAD:
            continue                                       # teachers disagree: a human has to settle that first
        current = float(getattr(cfg, knob.setting))
        direction = None
        if g["normal_share"] is not None and g["normal_share"] >= RAISE_IF_NORMAL_SHARE:
            direction, target = "less_sensitive", current + knob.step
        elif g["missed"] >= MIN_MISSED and (g["normal_share"] or 0.0) <= LOWER_IF_NORMAL_SHARE:
            direction, target = "more_sensitive", current - knob.step
        if direction is None:
            continue
        target = _round(knob, target)
        if target == _round(knob, current):
            continue                                       # already at the safe limit
        evidence = {"decided": g["decided"], "confirmed": g["confirmed"], "dismissed": g["dismissed"],
                    "normal_behaviour": g["normal_behaviour"], "bad_detection": g["bad_detection"],
                    "missed": g["missed"], "normal_share": g["normal_share"], "meaning": knob.meaning,
                    "teacher_spread": g["teacher_spread"]}
        created.append(store.add_suggestion({"room": cfg.room_name, "kind": kind, "setting": knob.setting,
                                             "current": current, "suggested": target, "direction": direction,
                                             "evidence": json.dumps(evidence)}))
    return created


def can_approve(cfg: Config, who: str) -> bool:
    """If EXAMGUARD_ADMIN_NAMES is set, only those names may approve; otherwise any signed-in teacher may.
    Names are typed at sign-in (there is one shared password), so this is a safeguard, not strong security."""
    admins = [a.strip().lower() for a in cfg.admin_names.split(",") if a.strip()]
    return not admins or (who or "").strip().lower() in admins


def approve(store: Store, cfg: Config, sid: int, actor: str) -> Dict:
    if not can_approve(cfg, actor):
        raise ReviewError("only an administrator listed in EXAMGUARD_ADMIN_NAMES can approve a change")
    s = store.get_suggestion(sid)
    if s is None:
        raise KeyError(sid)
    if s["setting"] not in ALLOWED_SETTINGS:
        raise ReviewError("this setting cannot be changed from here")
    out = store.decide_suggestion(sid, True, actor)
    overrides = read_overrides(cfg.data_path)
    overrides[s["setting"]] = s["suggested"]
    write_overrides(cfg.data_path, overrides)
    return out


def reject(store: Store, cfg: Config, sid: int, actor: str) -> Dict:
    if not can_approve(cfg, actor):
        raise ReviewError("only an administrator listed in EXAMGUARD_ADMIN_NAMES can reject a change")
    return store.decide_suggestion(sid, False, actor)


# ------------------------------------------------------------- overrides
def read_overrides(data_dir: Path) -> Dict[str, float]:
    """Approved changes (setting -> value). Unknown or out-of-range entries are ignored."""
    try:
        data = json.loads((Path(data_dir) / OVERRIDES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: Dict[str, float] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            knob = ALLOWED_SETTINGS.get(k)
            if knob and isinstance(v, (int, float)) and not isinstance(v, bool) and knob.lo <= v <= knob.hi:
                out[k] = int(v) if knob.integer else float(v)
    return out


def write_overrides(data_dir: Path, overrides: Dict[str, float]) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / OVERRIDES_FILE).write_text(json.dumps(overrides, indent=2, sort_keys=True), encoding="utf-8")


def reset_overrides(data_dir: Path) -> None:
    try:
        (Path(data_dir) / OVERRIDES_FILE).unlink()
    except OSError:
        pass
