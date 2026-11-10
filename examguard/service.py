"""Wires every component together (shared by run.py and the tests)."""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import List, Optional

from .clips import ClipRecorder
from .config import Config
from .demo import DemoWorld
from .hub import AlertHub
from .monitor import Monitor
from .notifier import EmailNotifier
from .marks import MarkWriter
from .roster import Roster
from .retention import RetentionWorker, apply_retention
from .seatmap import SeatAssigner, SeatMap
from .sources import describe_source, open_source
from .store import Store
from .vault import Vault

log = logging.getLogger("examguard.service")


@dataclass
class Services:
    cfg: Config
    store: Store
    vault: Vault
    hub: AlertHub
    notifier: EmailNotifier
    monitor: Monitor
    retention: RetentionWorker
    world: Optional[DemoWorld] = None
    roster: Optional[Roster] = None
    marks: Optional[MarkWriter] = None
    seatmap: Optional[SeatMap] = None
    monitors: Optional[List[Monitor]] = None      # every camera; monitors[0] is `monitor`
    worlds: Optional[List[DemoWorld]] = None      # demo mode only
    link: Optional[object] = None                 # edge / server role: the MQTT link (None when standalone)
    role: str = "standalone"

    @property
    def demo(self) -> bool:
        return self.world is not None

    def all_monitors(self) -> List[Monitor]:
        return self.monitors or [self.monitor]

    def start(self) -> None:
        self.retention.start()
        if self.link is not None:
            self.link.start()
        for m in self.all_monitors():
            m.start()

    def stop(self) -> None:
        for m in self.all_monitors():
            m.stop()                                  # an edge box flushes its last clips into the outbox here
        if self.link is not None:
            self.link.stop()
        self.retention.stop()
        self.store.close()


def ensure_password(cfg: Config) -> str:
    """Dashboard password: from config, else a stable random one saved in the data folder."""
    if cfg.password:
        return cfg.password
    f = cfg.data_path / "dashboard_password.txt"
    if f.exists():
        return f.read_text().strip()
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    pw = secrets.token_urlsafe(9)
    f.write_text(pw)
    try:
        f.chmod(0o600)
    except OSError:
        pass
    return pw


def build_services(cfg: Config, demo: bool = False, demo_auto: bool = True,
                   analyzer=None, detector=None, source=None, roster=None, identifier=None,
                   second=None) -> Services:
    """Wire everything together.

    ``second`` (tests only) is a dict with ``source``, ``analyzer`` and optionally ``detector`` that stands in
    for the optional second camera; real runs build it from ``cfg.second_source``.
    """
    role = (cfg.role or "standalone").strip().lower()
    if role not in ("standalone", "edge", "server"):
        raise ValueError(f"EXAMGUARD_ROLE must be standalone, edge or server (got '{cfg.role}')")
    if role == "server" and demo:
        raise ValueError("--demo cannot be combined with --role server (the server has no camera: use --role edge --demo "
                         "on the other computer)")
    wire = None
    if role != "standalone":
        from .splitmode import Wire
        wire = Wire(cfg.mqtt_key)                      # refuses to start without a valid shared key
    data = cfg.data_path
    data.mkdir(parents=True, exist_ok=True)
    vault = Vault(data / "evidence")
    store = Store(data / "examguard.db")
    hub = AlertHub()

    def thumb_loader(incident):
        names = store.media_names(incident["id"])
        return vault.get(names["thumb"]) if names and names["thumb"] else None

    notifier = EmailNotifier(cfg, thumb_loader)
    two_cameras = bool(cfg.second_source.strip()) or second is not None
    camera_names = [cfg.first_camera_name if two_cameras else "", cfg.second_camera_name if two_cameras else ""]

    link = None                                       # edge role: set below, read when a clip is finished

    def make_recorder(camera_name: str) -> ClipRecorder:
        def on_clip_done(event, clip, thumb):
            event.camera = event.camera or camera_name
            if role == "edge":                        # no local review queue: the server owns it
                link.submit_incident(event, clip, thumb)
                return
            incident = store.add_incident(event, cfg.room_name, clip, thumb)
            hub.publish("incident", incident)
            notifier.notify(incident)
        return ClipRecorder(cfg, vault, on_clip_done)

    if role == "server":
        from .splitmode import RemoteFeed, ServerLink
        roster = Roster.load(cfg.roster_file, cfg.roster_photos_dir) if cfg.roster_file else None
        seatmap = SeatMap(data / "seatmap.json")
        feed = RemoteFeed(cfg)
        link = ServerLink(cfg, wire, store, vault, hub, notifier, seatmap, feed)
        retention = RetentionWorker(cfg, store, vault)
        apply_retention(cfg, store, vault)
        marks = MarkWriter(roster, cfg.marks_column, cfg.mark_value) if roster is not None and cfg.write_marks and roster.path else None
        return Services(cfg, store, vault, hub, notifier, feed.monitors[0], retention, None, roster, marks, seatmap,
                        feed.monitors, None, link=link, role="server")

    worlds: List[DemoWorld] = []
    seatmap = SeatMap(None if demo else data / "seatmap.json")
    if role == "edge":
        from .splitmode import EdgeLink
        link = EdgeLink(cfg, wire, vault, seatmap)
    sources = []                      # one (source, analyzer, detector, label) per camera
    if demo:
        from .demo import CENTERS, demo_roster
        from .identify import FixedIdentifier
        from .seatmap import Seat, row_letters
        world = DemoWorld(cfg, auto=demo_auto)
        worlds.append(world)
        roster, mapping = demo_roster()
        demo_csv = data / "demo-roster.csv"             # a real file, so mark write-back can be tried
        with open(demo_csv, "w", encoding="utf-8-sig", newline="") as f:
            import csv as _csv
            w = _csv.writer(f)
            w.writerow(["name", "student_id", "mark"])
            for s in roster.students:
                w.writerow([s.name, s.student_id, ""])
        roster.path = str(demo_csv)
        identifier = FixedIdentifier(mapping)
        sources.append((world, world, (world if cfg.phone_enabled else None), "Simulated exam room (demo)"))
        seatmap.replace([Seat(f"{row_letters(i // 3)}{i % 3 + 1}", cx / 960.0, cy / 540.0, 0, roster.students[i].ref)
                         for i, (cx, cy) in enumerate(CENTERS)])
        if two_cameras:
            seatmap.replace(seatmap.seats() + [Seat(f"{row_letters(i // 3 + 2)}{i % 3 + 1}", cx / 960.0, cy / 540.0, 1)
                                               for i, (cx, cy) in enumerate(CENTERS)])
            world2 = DemoWorld(cfg, auto=demo_auto, seed=11)
            worlds.append(world2)
            sources.append((world2, world2, (world2 if cfg.phone_enabled else None), "Simulated exam room 2 (demo)"))
    else:
        if analyzer is None:
            from .vision import load_detectors, load_face_analyzer
            analyzer = load_face_analyzer(cfg)
            detector = load_detectors(cfg)
        if source is None:
            source = open_source(cfg.source, cfg.loop_video)
        sources.append((source, analyzer, detector, describe_source(cfg.source)))
        if two_cameras:
            if second is not None:
                sources.append((second["source"], second["analyzer"], second.get("detector"), "second camera (test)"))
            else:
                from .vision import load_detectors, load_face_analyzer
                sources.append((open_source(cfg.second_source, cfg.loop_video), load_face_analyzer(cfg),
                                load_detectors(cfg), describe_source(cfg.second_source)))

    identifiers = [identifier]
    if roster is None and cfg.roster_file:
        from .identify import Identifier, SFaceEmbedder
        roster = Roster.load(cfg.roster_file, cfg.roster_photos_dir)
        identifier = Identifier(cfg, roster, SFaceEmbedder(cfg.models_path))
        for w in identifier.enroll():
            log.warning("roster: %s", w)
        log.info("roster loaded: %d students, %d with a usable face photo", len(roster),
                 len(identifier._enrolled))
        identifiers = [identifier]
    if len(sources) > 1:
        # every camera needs its own tracker state, but the enrolled photos are shared
        first = identifiers[0]
        if first is not None and hasattr(first, "for_another_camera"):
            identifiers.append(first.for_another_camera())
        else:
            identifiers.append(first)        # demo: fixed identities are the same for both views

    monitors: List[Monitor] = []
    for i, (src, an, det, label) in enumerate(sources):
        monitors.append(Monitor(
            cfg, src, an, det, make_recorder(camera_names[i]), label,
            identifiers[i] if i < len(identifiers) else None, camera_name=camera_names[i],
            seat_assigner=SeatAssigner(seatmap, i, cfg.seat_radius_faces), roster=roster, camera_index=i))
    retention = RetentionWorker(cfg, store, vault)
    apply_retention(cfg, store, vault)
    marks = None
    if roster is not None and cfg.write_marks and roster.path:
        marks = MarkWriter(roster, cfg.marks_column, cfg.mark_value)
    if link is not None:
        link.monitors = monitors
    return Services(cfg, store, vault, hub, notifier, monitors[0], retention, worlds[0] if worlds else None,
                    roster, marks, seatmap, monitors, worlds or None, link=link, role=role)
