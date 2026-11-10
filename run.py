#!/usr/bin/env python3
"""Start ExamGuard.

    python run.py --demo                 # no camera needed: simulated exam room
    python run.py                        # laptop webcam
    python run.py --source http://192.168.1.50:4747/video   # phone camera (DroidCam)
    python run.py --source exam.mp4      # a recorded video
    python run.py --profile gpu          # performance profile: laptop (default) | edge | jetson | gpu
    python run.py --second-source http://192.168.1.51:4747/video   # optional 2nd camera for rooms over ~100 students
    python run.py --role edge            # camera + AI box in the room, sends alerts to the server over MQTT
    python run.py --role server          # teachers' dashboard that receives them (see README "Edge box + server")

For automatic restarts and the "Apply and restart" button, start it with:  python start.py  (or start.bat / start.sh)
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from examguard.config import Config
from examguard.service import build_services, ensure_password
from examguard.web.app import create_app


def main() -> int:
    ap = argparse.ArgumentParser(description="ExamGuard: AIoT classroom exam integrity monitoring (Phase 1 MVP)")
    ap.add_argument("--demo", action="store_true", help="simulated room, no camera or AI models needed")
    ap.add_argument("--source", help="webcam index (0), stream URL, or video file (default: 0)")
    ap.add_argument("--room", help="room name shown on the dashboard and in clips")
    ap.add_argument("--host", help="0.0.0.0 lets other devices on the network open the dashboard")
    ap.add_argument("--port", type=int)
    ap.add_argument("--https", action="store_true", help="serve the dashboard over HTTPS (self-signed certificate)")
    ap.add_argument("--no-phone", action="store_true", help="turn off YOLO phone detection")
    ap.add_argument("--profile", choices=["laptop", "edge", "jetson", "gpu"],
                    help="performance profile (default: the one chosen on the System check page, else laptop). "
                         "edge, jetson and gpu are untested starting points")
    ap.add_argument("--second-source", help="OPTIONAL second camera (other corner) for rooms over about 100 students")
    ap.add_argument("--role", choices=["standalone", "edge", "server"],
                    help="standalone = everything on this computer (default); edge = camera + AI box that sends alerts "
                         "through an MQTT broker; server = dashboard that receives them. edge/server are untested on real hardware")
    ap.add_argument("--node", help="a name for this edge box")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    try:
        cfg = Config.from_env(source=args.source, room_name=args.room, host=args.host, port=args.port,
                              phone_enabled=False if args.no_phone else None, https=True if args.https else None,
                              profile=args.profile, second_source=args.second_source,
                              role=args.role, node_name=args.node)
    except ValueError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    if cfg.profile != "laptop":
        logging.getLogger("examguard").warning(
            "performance profile '%s' has NOT been tested on real hardware: check the System check page, "
            "scripts/tune.py and scripts/evaluate.py before relying on it", cfg.profile)
    try:
        svc = build_services(cfg, demo=args.demo)
    except RuntimeError as exc:                      # e.g. face model could not be downloaded
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1

    if cfg.role == "edge":
        return run_edge(cfg, svc, args.demo)
    app = create_app(svc)
    svc.start()
    if os.environ.get("EXAMGUARD_SUPERVISED") and cfg.role != "server":     # restarting the server cannot fix a stuck camera
        from examguard.watchdog import Watchdog

        def _restart(reason: str) -> None:
            os._exit(75)                              # start.py starts us again
        Watchdog(cfg.watchdog_seconds, svc.all_monitors(), _restart).start()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    ssl_ctx = None
    if cfg.https:
        from examguard.tls import ensure_self_signed
        if cfg.tls_cert and cfg.tls_key:
            ssl_ctx = (cfg.tls_cert, cfg.tls_key)
        else:
            ssl_ctx = tuple(str(p) for p in ensure_self_signed(cfg.data_path / "tls"))
    from examguard.tls import primary_ip
    local = cfg.host in ("0.0.0.0", "127.0.0.1")
    url = cfg.public_url or f"{cfg.scheme}://{'localhost' if local else cfg.host}:{cfg.port}"
    lan_url = f"{cfg.scheme}://{primary_ip()}:{cfg.port}" if cfg.host == "0.0.0.0" else ""
    print("\n" + "=" * 62)
    print(f"  ExamGuard is running{'  (DEMO MODE - simulated room)' if args.demo else ''}")
    print(f"  Dashboard : {url}")
    if lan_url:
        print(f"  From phones/other devices on this network: {lan_url}")
    if cfg.https and not (cfg.tls_cert and cfg.tls_key):
        print("  HTTPS     : self-signed certificate. Each browser shows a one-time warning: accept it once.")
    print(f"  Profile   : {cfg.profile}" + ("" if cfg.profile == "laptop" else "  (NOT tested on real hardware)"))
    if cfg.role == "server":
        print(f"  Role      : SERVER: receives alerts from the edge box through {cfg.mqtt_host}:{cfg.mqtt_port} (NOT tested on real hardware)")
    if len(svc.all_monitors()) > 1:
        print("  Cameras   : 2 (front + back); both feed the same incident list")
    print(f"  Password  : {ensure_password(cfg)}")
    print(f"  Alerts    : browser + sound{', email' if svc.notifier.enabled else ''}")
    print("  Press Ctrl+C to stop.")
    print("=" * 62 + "\n")
    try:
        app.run(host=cfg.host, port=cfg.port, threaded=True, use_reloader=False,
                ssl_context=ssl_ctx, load_dotenv=False)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping ...")
        svc.stop()
    return 0


def run_edge(cfg, svc, demo: bool) -> int:
    """The edge box has no dashboard: it watches, records, and sends alerts to the server."""
    svc.start()
    if os.environ.get("EXAMGUARD_SUPERVISED"):
        from examguard.watchdog import Watchdog
        Watchdog(cfg.watchdog_seconds, svc.all_monitors(), lambda reason: os._exit(75)).start()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    print("\n" + "=" * 62)
    print(f"  ExamGuard EDGE box '{svc.link.node}'{'  (DEMO MODE - simulated room)' if demo else ''}")
    print(f"  Room      : {cfg.room_name}")
    print(f"  Broker    : {cfg.mqtt_host}:{cfg.mqtt_port}{' (TLS)' if cfg.mqtt_tls else ''}")
    print(f"  Profile   : {cfg.profile}" + ("" if cfg.profile == "laptop" else "  (NOT tested on real hardware)"))
    print("  Alerts go to the server's review queue; they are kept in an outbox here until delivered.")
    print("  NOT tested on real hardware or a real broker. Press Ctrl+C to stop.")
    print("=" * 62 + "\n")
    try:
        while True:
            signal.pause() if hasattr(signal, "pause") else __import__("time").sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping ...")
        svc.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
