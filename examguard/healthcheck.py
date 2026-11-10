"""Pre-exam health check: one list of green / yellow / red lights for everything ExamGuard depends on.

Shown on the dashboard's "System check" page and at /api/health. It only *reads* state: it never changes anything.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List

from .profiles import describe

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"
MB = 1024 * 1024


def _item(level: str, title: str, detail: str, advice: str = "") -> Dict[str, str]:
    return {"level": level, "title": title, "detail": detail, "advice": advice}


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _edge_items(svc) -> List[Dict[str, str]]:
    from . import splitmode
    from .splitmode import MAX_CLOCK_SKEW
    info = svc.link.info()
    items: List[Dict[str, str]] = []
    if not info["connected"]:
        items.append(_item(FAIL, "Link to the broker", f"not connected ({info['last_error'] or 'trying'})",
                           "Check that the MQTT broker (Mosquitto) is running, EXAMGUARD_MQTT_HOST/PORT/USER/PASSWORD are right "
                           "and this computer can reach it. Alerts wait safely on the edge box meanwhile."))
    else:
        items.append(_item(OK, "Link to the broker", f"connected to {svc.cfg.mqtt_host}:{svc.cfg.mqtt_port}"))
    age = info["edge_age"]
    if age is None:
        items.append(_item(FAIL, "Edge box", "nothing received from it yet",
                           "Start it with: python run.py --role edge. It needs the same EXAMGUARD_MQTT_KEY, room name and broker."))
    elif age > splitmode.OFFLINE_AFTER:
        items.append(_item(FAIL, "Edge box", f"no report for {age:.0f} s: it is off, offline or stopped",
                           "Check the box has power and network, and that ExamGuard is running on it. Alerts it already "
                           "recorded are kept there and arrive when it reconnects."))
    else:
        items.append(_item(OK, "Edge box", f"'{', '.join(info['nodes']) or '?'}' reported {age:.0f} s ago"
                                         + (f", profile {info['edge_profile']}" if info.get("edge_profile") else "")))
        if abs(info["skew"]) > MAX_CLOCK_SKEW:
            items.append(_item(WARN, "Clocks", f"the edge box's clock differs from this computer's by {info['skew']:+.0f} s",
                               "Turn on automatic time (NTP) on both: alert times come from the edge box."))
        backlog = info.get("outbox") or 0
        if backlog:
            items.append(_item(WARN, "Alerts waiting on the edge box", f"{backlog} not delivered yet",
                               "They are sent as soon as the link works." if info["connected"] else "The link is down."))
        if info.get("dropped"):
            items.append(_item(FAIL, "Alerts lost", f"{info['dropped']} alert(s) were dropped on the edge box (its outbox filled up)",
                               "The server was unreachable for too long. Raise EXAMGUARD_OUTBOX_MAX_MB and fix the link."))
        if len(info["nodes"]) > 1:
            items.append(_item(WARN, "Several edge boxes", f"{', '.join(info['nodes'])} all report for this room",
                               "One edge box per room is supported. Use a different room name for each."))
    if info["rejected"]:
        items.append(_item(WARN, "Rejected messages", f"{info['rejected']} message(s) could not be accepted",
                           "Usually EXAMGUARD_MQTT_KEY differs between the computers. See the audit trail."))
    items.append(_item(INFO, "Live view", "in this mode the live picture is a slow slideshow (about one picture every "
                                         f"{svc.cfg.live_seconds:g} s) sent by the edge box"))
    return items


def run_checks(svc) -> Dict:
    cfg = svc.cfg
    items: List[Dict[str, str]] = []

    # -- profile
    prof = describe(cfg.profile)
    if prof["tested"]:
        items.append(_item(OK, "Performance profile", f"{prof['title']} (the profile ExamGuard was developed on)"))
    else:
        items.append(_item(WARN, "Performance profile",
                           f"{prof['title']}: this profile has NOT been tested on real hardware",
                           "Run scripts/tune.py and scripts/evaluate.py on this computer and check the speed below."))

    # -- edge box link (server role)
    if svc.role == "server" and svc.link is not None:
        items.extend(_edge_items(svc))

    # -- cameras
    for m in svc.all_monitors():
        st = m.status()
        name = st.get("camera") or "Camera"
        if st.get("remote") and st["state"] == "offline":
            continue                                   # already reported above as the edge box being offline
        src_state = st.get("source_state") or ""
        age = st.get("last_frame_age")
        if st["state"] == "running" and src_state in ("connected", "simulation", "playing"):
            items.append(_item(OK, f"{name}: video", f"{st['source']} is {src_state}, last picture {age if age is not None else '-'} s ago"))
        elif st["state"] == "error":
            items.append(_item(FAIL, f"{name}: video", f"processing error: {st.get('error') or 'unknown'}",
                               "Check the terminal for details, then restart ExamGuard."))
        else:
            items.append(_item(FAIL, f"{name}: video", f"{st['source']} is {src_state or st['state']}",
                               "Check the cable or Wi-Fi, that no other app is using the camera, and the --source address."))
        target = cfg.process_fps
        if st["state"] == "running":
            if st["fps"] < 0.5 * target:
                items.append(_item(WARN, f"{name}: speed", f"{st['fps']} frames/s analysed (target {target:g})",
                                   "Too slow to catch short behaviour reliably. Try fewer faces per camera, "
                                   "--no-phone, a smaller EXAMGUARD_MAX_WIDTH, or a faster computer."))
            else:
                items.append(_item(OK, f"{name}: speed", f"{st['fps']} frames/s analysed (target {target:g})"))
        n, fh = st["students"], st.get("face_health") or {}
        if fh.get("advice"):
            items.append(_item(WARN, f"{name}: face size", fh["advice"]))
        elif n:
            items.append(_item(OK, f"{name}: faces", f"{n} faces found, typical width {fh.get('median_px')} px"))
        else:
            items.append(_item(WARN, f"{name}: faces", "no faces found right now",
                               "Seat a few people in view, or check the camera angle and lighting."))
        if n >= 0.9 * cfg.max_students:
            items.append(_item(WARN, f"{name}: crowded", f"{n} faces is close to the limit of {cfg.max_students} per camera",
                               "For more than about 100 students aim a second camera at the other part of the room "
                               "(EXAMGUARD_SECOND_SOURCE) or raise EXAMGUARD_MAX_STUDENTS."))

    # -- detectors
    mon = svc.monitor.status()
    items.append(_item(OK if mon.get("phone_detector") else INFO, "Phone detection",
                       "on" if mon.get("phone_detector") else "off (phones will not be flagged)"))
    items.append(_item(OK if mon.get("hand_detector") else INFO, "Passing-an-object detection",
                       "on" if mon.get("hand_detector") else "off"))

    # -- storage
    data = cfg.data_path
    try:
        free = shutil.disk_usage(data if data.exists() else data.parent).free
        if free < 300 * MB:
            items.append(_item(FAIL, "Disk space", f"only {free // MB} MB free", "Free some space: evidence clips could not be saved."))
        elif free < 2048 * MB:
            items.append(_item(WARN, "Disk space", f"{free // MB} MB free", "Free some space before a long exam."))
        else:
            items.append(_item(OK, "Disk space", f"{free // MB} MB free"))
    except OSError as exc:
        items.append(_item(WARN, "Disk space", f"could not be read ({exc})"))
    items.append(_item(INFO, "Stored evidence", f"{_dir_size(data / 'evidence') // MB} MB (deleted automatically by the retention rules)"))
    try:
        from .clips import _ffmpeg_exe
        ok = os.path.exists(_ffmpeg_exe())
        items.append(_item(OK if ok else FAIL, "Video encoder (ffmpeg)", "found" if ok else "missing",
                           "" if ok else "Run: pip install -r requirements.txt   (imageio-ffmpeg). Without it no evidence clips are made."))
    except Exception as exc:
        items.append(_item(FAIL, "Video encoder (ffmpeg)", f"not available ({exc})", "Run: pip install -r requirements.txt"))

    # -- people and seats
    if svc.roster is not None:
        n = len(svc.roster)
        with_photo = sum(1 for s in svc.roster.students if s.photo_bgr is not None)
        level = OK if with_photo == n and n else WARN
        items.append(_item(level, "Student roster", f"{n} students, {with_photo} with a photo",
                           "" if level == OK else "Students without a photo are never suggested by face matching. Run scripts/check_roster.py."))
        if svc.marks is not None and svc.roster.path:
            writable = os.access(svc.roster.path, os.W_OK)
            items.append(_item(OK if writable else FAIL, "Spreadsheet marks",
                               f"confirmed incidents write '{cfg.mark_value}' into {Path(svc.roster.path).name}"
                               if writable else "the roster file cannot be written",
                               "" if writable else "Check the file's permissions and that it is closed in Excel."))
    else:
        items.append(_item(INFO, "Student roster", "none loaded: students appear as 'Student N'"))
    if svc.seatmap is not None:
        n_seats = len(svc.seatmap)
        if n_seats:
            items.append(_item(OK, "Seat map", f"{n_seats} seats defined"))
        else:
            items.append(_item(INFO, "Seat map", "no seats defined: alerts will not say where the student sits",
                               "Open 'Seat map' and mark the seats once for this room."))

    # -- alerts and security
    items.append(_item(OK if svc.notifier.enabled else INFO, "Email alerts",
                       "on" if svc.notifier.enabled else "off (the dashboard still alerts with sound)"))
    if cfg.host == "0.0.0.0" and not cfg.https:
        items.append(_item(WARN, "Dashboard security", "open to the network without HTTPS",
                           "Start with --https so the password and video cannot be read on the Wi-Fi."))
    else:
        items.append(_item(OK, "Dashboard security", "HTTPS on" if cfg.https else "only this computer can open the dashboard"))

    worst = FAIL if any(i["level"] == FAIL for i in items) else WARN if any(i["level"] == WARN for i in items) else OK
    return {"overall": worst, "items": items, "profile": cfg.profile}
