"""SQLite incident log + audit trail.

Status rules (the human-in-the-loop safeguard lives here):
  * every incident is created as ``pending``
  * the ONLY code path that changes status is :meth:`Store.review`, which needs a named reviewer
  * every create / review / view / export / purge is written to the audit table
"""
from __future__ import annotations

import csv
import io
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .models import DISMISS_REASONS, KIND_LABELS, Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts  REAL NOT NULL,
    room        TEXT NOT NULL,
    track_id    INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    severity    TEXT NOT NULL,
    score       REAL NOT NULL,
    details     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    reviewer    TEXT,
    reviewed_ts REAL,
    note        TEXT,
    clip_name   TEXT,
    thumb_name  TEXT,
    purged_ts   REAL,
    student_name TEXT,
    student_ref  TEXT,
    match_score  REAL,
    id_corrected INTEGER NOT NULL DEFAULT 0,
    seat         TEXT,
    seat_student TEXT,
    id_source    TEXT,
    camera       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status, created_ts);
CREATE TABLE IF NOT EXISTS exams (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    room        TEXT NOT NULL,
    started_ts  REAL NOT NULL,
    ended_ts    REAL,
    started_by  TEXT NOT NULL,
    ended_by    TEXT
);
CREATE TABLE IF NOT EXISTS missed_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_id     INTEGER,
    room        TEXT NOT NULL,
    ts          REAL NOT NULL,
    reported_ts REAL NOT NULL,
    kind        TEXT NOT NULL,
    reported_by TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS suggestions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts  REAL NOT NULL,
    room        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    setting     TEXT NOT NULL,
    current_val REAL NOT NULL,
    suggested   REAL NOT NULL,
    direction   TEXT NOT NULL,
    evidence    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    decided_by  TEXT,
    decided_ts  REAL
);
CREATE TABLE IF NOT EXISTS audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    incident_id INTEGER,
    detail      TEXT NOT NULL DEFAULT ''
);
"""

STATUSES = ("pending", "confirmed", "dismissed", "expired")


class ReviewError(ValueError):
    pass


def _identity_note(d: Dict) -> str:
    """Tell the teacher where a suggested name came from, and shout when the two sources disagree."""
    if not d["student_name"]:
        return ""
    if d["id_corrected"]:
        return "Name chosen by a teacher"
    seat = d["seat"] or "?"
    if d["id_source"] == "seat":
        return (f"Name comes from the SEAT MAP (who is assigned to seat {seat}), not from the face: "
                f"check it against the video")
    if d["seat_student"] and d["seat_student"] != d["student_name"]:
        return (f"Face matching suggests {d['student_name']}, but the seat map says {d['seat_student']} sits in "
                f"seat {seat}. Check carefully who it is")
    return "Name SUGGESTED by face matching: check it against the video"


def serialize(row: sqlite3.Row) -> Dict:
    d = dict(row)
    return {
        "id": d["id"],
        "ts": d["created_ts"],
        "room": d["room"],
        "track_id": d["track_id"],
        "student": d["student_name"] or f"Student {d['track_id']}" + (f" ({d['camera']})" if d["camera"] else ""),
        "student_label": f"Student {d['track_id']}" + (f" ({d['camera']})" if d["camera"] else ""),
        "seat": d["seat"],
        "camera": d["camera"] or "",
        "id_source": d["id_source"],
        "student_name": d["student_name"],
        "student_ref": d["student_ref"],
        "match_score": d["match_score"],
        "id_corrected": bool(d["id_corrected"]),
        "identity_note": _identity_note(d),
        "kind": d["kind"],
        "kind_label": KIND_LABELS.get(d["kind"], d["kind"]),
        "severity": d["severity"],
        "score": round(d["score"], 2),
        "details": d["details"],
        "status": d["status"],
        "reviewer": d["reviewer"],
        "reviewed_ts": d["reviewed_ts"],
        "note": d["note"] or "",
        "has_clip": bool(d["clip_name"]),
        "has_thumb": bool(d["thumb_name"]),
        "purged": d["purged_ts"] is not None,
        "dismiss_reason": d["dismiss_reason"],
        "dismiss_reason_label": DISMISS_REASONS.get(d["dismiss_reason"] or "", ""),
        "ignored": d["status"] == "pending" and d["ignored_ts"] is not None,
        "ignored_by": d["ignored_by"] if d["status"] == "pending" else None,
        "exam_id": d["exam_id"],
    }


class Store:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            have = {r[1] for r in self._db.execute("PRAGMA table_info(incidents)")}
            for col, ddl in (("student_name", "TEXT"), ("student_ref", "TEXT"), ("match_score", "REAL"),
                             ("id_corrected", "INTEGER NOT NULL DEFAULT 0"), ("seat", "TEXT"),
                             ("seat_student", "TEXT"), ("id_source", "TEXT"), ("camera", "TEXT NOT NULL DEFAULT ''"),
                             ("dismiss_reason", "TEXT"), ("ignored_ts", "REAL"), ("ignored_by", "TEXT"),
                             ("exam_id", "INTEGER"), ("remote_id", "TEXT")):
                if col not in have:                       # databases created by earlier versions
                    self._db.execute(f"ALTER TABLE incidents ADD COLUMN {col} {ddl}")
            # alerts that arrive from an edge box carry a unique id, so a repeat can never be saved twice
            self._db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_incidents_remote ON incidents(remote_id) "
                             "WHERE remote_id IS NOT NULL")
            self._db.commit()

    # ------------------------------------------------------------ incidents
    def has_remote_id(self, remote_id: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM incidents WHERE remote_id=?", (remote_id,)).fetchone() is not None

    def add_incident(self, event: Event, room: str, clip: Optional[str], thumb: Optional[str],
                     remote_id: Optional[str] = None) -> Dict:
        with self._lock:
            exam_id = self._open_exam(room, event.ts)["id"]
            cur = self._db.execute(
                "INSERT INTO incidents (created_ts, room, track_id, kind, severity, score, details,"
                " clip_name, thumb_name, student_name, student_ref, match_score,"
                " seat, seat_student, id_source, camera, exam_id, remote_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (event.ts, room, event.track_id, event.kind, event.severity, event.score,
                 event.details, clip, thumb, event.student_name, event.student_ref, event.match_score,
                 event.seat, event.seat_student, event.id_source, event.camera or "", exam_id, remote_id),
            )
            iid = cur.lastrowid
            self._audit("system", "incident_created", iid, f"{event.kind} / {event.label}")
            self._db.commit()
        return self.get(iid)

    def get(self, incident_id: int) -> Optional[Dict]:
        row = self._row(incident_id)
        return serialize(row) if row else None

    def _row(self, incident_id: int) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()

    def media_names(self, incident_id: int) -> Optional[Dict[str, Optional[str]]]:
        row = self._row(incident_id)
        return {"clip": row["clip_name"], "thumb": row["thumb_name"]} if row else None

    def list(self, status: Optional[str] = None, limit: int = 200) -> List[Dict]:
        """status: one of STATUSES, 'history' (everything already reviewed or expired), or None (all)."""
        with self._lock:
            if status == "history":
                rows = self._db.execute(
                    "SELECT * FROM incidents WHERE status != 'pending' ORDER BY created_ts DESC LIMIT ?",
                    (limit,)).fetchall()
            elif status in STATUSES:
                order = "DESC" if status != "pending" else "ASC"      # oldest pending first
                first = "(ignored_ts IS NOT NULL), " if status == "pending" else ""   # "Ignore"d ones go last
                rows = self._db.execute(
                    f"SELECT * FROM incidents WHERE status=? ORDER BY {first}created_ts {order} LIMIT ?",
                    (status, limit)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM incidents ORDER BY created_ts DESC LIMIT ?", (limit,)).fetchall()
        return [serialize(r) for r in rows]

    def count_confirmed_for(self, ref: str) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM incidents WHERE status='confirmed' AND student_ref=?", (ref,)).fetchone()[0]

    def count(self, status: str) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM incidents WHERE status=?", (status,)).fetchone()[0]

    # --------------------------------------------------------------- review
    def review(self, incident_id: int, decision: str, reviewer: str, note: str = "",
               reason: Optional[str] = None) -> Dict:
        """The only way an incident leaves ``pending``. Always done by a named human.

        ``reason`` (dismissals only, optional here, required by the dashboard) is why it was a false alarm.
        """
        reviewer = (reviewer or "").strip()
        if not reviewer:
            raise ReviewError("a reviewer name is required")
        if decision not in ("confirm", "dismiss"):
            raise ReviewError("decision must be 'confirm' or 'dismiss'")
        new_status = "confirmed" if decision == "confirm" else "dismissed"
        note = (note or "").strip()[:500]
        reason = (reason or "").strip() or None
        if reason is not None and (decision != "dismiss" or reason not in DISMISS_REASONS):
            raise ReviewError("unknown dismiss reason" if decision == "dismiss" else "only a dismissal has a reason")
        with self._lock:
            row = self._row(incident_id)
            if row is None:
                raise KeyError(incident_id)
            if row["status"] == "expired":
                raise ReviewError("this alert expired unreviewed and its evidence was deleted")
            if new_status == "confirmed" and not row["clip_name"] and row["purged_ts"] is not None:
                raise ReviewError("the evidence was already deleted, so it cannot be confirmed")
            self._db.execute(
                "UPDATE incidents SET status=?, reviewer=?, reviewed_ts=?, note=?, dismiss_reason=?,"
                " ignored_ts=NULL, ignored_by=NULL WHERE id=?",
                (new_status, reviewer, time.time(), note, reason, incident_id))
            self._audit(reviewer, f"review_{new_status}", incident_id,
                        f"was {row['status']}" + (f"; reason: {reason}" if reason else "")
                        + (f"; note: {note}" if note else ""))
            self._db.commit()
        return self.get(incident_id)

    def set_student(self, incident_id: int, ref: Optional[str], name: Optional[str], actor: str) -> Dict:
        """A teacher corrects (or clears) who the flagged student is."""
        with self._lock:
            row = self._row(incident_id)
            if row is None:
                raise KeyError(incident_id)
            self._db.execute(
                "UPDATE incidents SET student_ref=?, student_name=?, id_corrected=1 WHERE id=?",
                (ref, name, incident_id))
            self._audit(actor, "student_corrected", incident_id,
                        f"{row['student_name'] or 'unidentified'} -> {name or 'unidentified'}")
            self._db.commit()
        return self.get(incident_id)

    # ------------------------------------------------- ignore ("not deciding now")
    def ignore(self, incident_id: int, actor: str, on: bool = True) -> Dict:
        """Park a pending alert. It stays pending (a human still has to decide) and is NEVER counted as a
        false alarm; it just drops to the bottom of the queue."""
        actor = (actor or "").strip()
        if not actor:
            raise ReviewError("a reviewer name is required")
        with self._lock:
            row = self._row(incident_id)
            if row is None:
                raise KeyError(incident_id)
            if row["status"] != "pending":
                raise ReviewError("only an alert that is still waiting for review can be ignored")
            if on:
                self._db.execute("UPDATE incidents SET ignored_ts=?, ignored_by=? WHERE id=?",
                                 (time.time(), actor, incident_id))
            else:
                self._db.execute("UPDATE incidents SET ignored_ts=NULL, ignored_by=NULL WHERE id=?", (incident_id,))
            self._audit(actor, "ignored" if on else "unignored", incident_id)
            self._db.commit()
        return self.get(incident_id)

    def set_dismiss_reason(self, incident_id: int, reason: str, actor: str) -> Dict:
        """Correct the reason given for a dismissal (for example after the exam, when the teacher has time)."""
        actor = (actor or "").strip()
        if not actor:
            raise ReviewError("a reviewer name is required")
        if reason not in DISMISS_REASONS:
            raise ReviewError("unknown dismiss reason")
        with self._lock:
            row = self._row(incident_id)
            if row is None:
                raise KeyError(incident_id)
            if row["status"] != "dismissed":
                raise ReviewError("only a dismissed alert has a reason")
            self._db.execute("UPDATE incidents SET dismiss_reason=? WHERE id=?", (reason, incident_id))
            self._audit(actor, "reason_corrected", incident_id, f"{row['dismiss_reason'] or 'none'} -> {reason}")
            self._db.commit()
        return self.get(incident_id)

    # ------------------------------------------------------------------ exams
    def _open_exam(self, room: str, now: float) -> sqlite3.Row:
        """The exam that is running in this room; starts one when needed (the first alert or missed report).
        A room that has been silent for more than 6 hours is treated as a new exam."""
        row = self._db.execute(
            "SELECT * FROM exams WHERE room=? AND ended_ts IS NULL ORDER BY id DESC LIMIT 1", (room,)).fetchone()
        if row is not None:
            last = self._db.execute(
                "SELECT MAX(ts) FROM (SELECT MAX(created_ts) AS ts FROM incidents WHERE exam_id=? "
                "UNION ALL SELECT MAX(ts) FROM missed_events WHERE exam_id=?)", (row["id"], row["id"])).fetchone()[0]
            last = max(row["started_ts"], last or 0.0)
            if now - last <= 6 * 3600:
                return row
            self._db.execute("UPDATE exams SET ended_ts=?, ended_by='auto (6 h of silence)' WHERE id=?", (last, row["id"]))
            self._audit("system", "exam_auto_ended", None, f"exam {row['id']}")
        cur = self._db.execute("INSERT INTO exams (room, started_ts, started_by) VALUES (?,?,?)",
                               (room, now, "auto (first alert)"))
        self._audit("system", "exam_started", None, f"exam {cur.lastrowid}, {room}")
        return self._db.execute("SELECT * FROM exams WHERE id=?", (cur.lastrowid,)).fetchone()

    def start_exam(self, room: str, actor: str) -> Dict:
        """A teacher starts the exam explicitly (optional: the first alert starts one by itself)."""
        actor = (actor or "").strip()
        if not actor:
            raise ReviewError("a reviewer name is required")
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM exams WHERE room=? AND ended_ts IS NULL", (room,)).fetchone()
            if row is not None:
                raise ReviewError("an exam is already running in this room")
            cur = self._db.execute("INSERT INTO exams (room, started_ts, started_by) VALUES (?,?,?)",
                                   (room, time.time(), actor))
            self._audit(actor, "exam_started", None, f"exam {cur.lastrowid}, {room}")
            self._db.commit()
        return self.exam_summary(cur.lastrowid)

    def current_exam(self, room: str) -> Optional[Dict]:
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM exams WHERE room=? AND ended_ts IS NULL ORDER BY id DESC LIMIT 1", (room,)).fetchone()
        return self.exam_summary(row["id"]) if row else None

    def finish_exam(self, room: str, actor: str) -> Dict:
        actor = (actor or "").strip()
        if not actor:
            raise ReviewError("a reviewer name is required")
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM exams WHERE room=? AND ended_ts IS NULL ORDER BY id DESC LIMIT 1", (room,)).fetchone()
            if row is None:
                raise ReviewError("no exam is running in this room")
            self._db.execute("UPDATE exams SET ended_ts=?, ended_by=? WHERE id=?", (time.time(), actor, row["id"]))
            self._audit(actor, "exam_finished", None, f"exam {row['id']}")
            self._db.commit()
        return self.exam_summary(row["id"])

    def list_exams(self, room: Optional[str] = None, limit: int = 30) -> List[Dict]:
        with self._lock:
            if room:
                rows = self._db.execute("SELECT * FROM exams WHERE room=? ORDER BY id DESC LIMIT ?", (room, limit)).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM exams ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r["id"], "room": r["room"], "started": r["started_ts"], "ended": r["ended_ts"],
                 "started_by": r["started_by"], "ended_by": r["ended_by"], "open": r["ended_ts"] is None} for r in rows]

    def exam_summary(self, exam_id: int) -> Optional[Dict]:
        with self._lock:
            ex = self._db.execute("SELECT * FROM exams WHERE id=?", (exam_id,)).fetchone()
            if ex is None:
                return None
            rows = self._db.execute(
                "SELECT * FROM incidents WHERE exam_id=? ORDER BY created_ts", (exam_id,)).fetchall()
            missed = self._db.execute(
                "SELECT * FROM missed_events WHERE exam_id=? ORDER BY ts", (exam_id,)).fetchall()
        alerts = [serialize(r) for r in rows]
        counts = {"alerts": len(alerts), "confirmed": 0, "dismissed": 0, "waiting": 0, "ignored": 0, "expired": 0,
                  "missed": len(missed)}
        reasons: Dict[str, int] = {}
        for a in alerts:
            if a["status"] == "pending":
                counts["waiting"] += 1
                counts["ignored"] += 1 if a["ignored"] else 0
            else:
                counts[a["status"]] += 1
            if a["status"] == "dismissed":
                key = a["dismiss_reason"] or "none"
                reasons[key] = reasons.get(key, 0) + 1
        slim = [{k: a[k] for k in ("id", "ts", "student", "seat", "kind", "kind_label", "status", "dismiss_reason",
                                   "dismiss_reason_label", "reviewer", "ignored", "details")} for a in alerts]
        return {
            "exam": {"id": ex["id"], "room": ex["room"], "started": ex["started_ts"], "ended": ex["ended_ts"],
                     "started_by": ex["started_by"], "ended_by": ex["ended_by"], "open": ex["ended_ts"] is None},
            "counts": counts, "reasons": reasons, "alerts": slim,
            "missed": [{"id": m["id"], "ts": m["ts"], "kind": m["kind"],
                        "kind_label": KIND_LABELS.get(m["kind"], "Something else"),
                        "reported_by": m["reported_by"], "note": m["note"]} for m in missed],
        }

    # ------------------------------------------------------- missed events
    def add_missed(self, room: str, kind: str, actor: str, note: str = "", minutes_ago: float = 0.0) -> Dict:
        """A teacher reports something ExamGuard did NOT flag. Only the rule, the time and a short note are kept."""
        actor = (actor or "").strip()
        if not actor:
            raise ReviewError("a reviewer name is required")
        if kind != "other" and kind not in KIND_LABELS:
            raise ReviewError("unknown kind of behaviour")
        try:
            minutes_ago = float(minutes_ago or 0.0)
        except (TypeError, ValueError):
            raise ReviewError("minutes ago must be a number")
        if not 0.0 <= minutes_ago <= 180.0:
            raise ReviewError("minutes ago must be between 0 and 180")
        note = (note or "").strip()[:200]
        now = time.time()
        with self._lock:
            exam_id = self._open_exam(room, now)["id"]
            cur = self._db.execute(
                "INSERT INTO missed_events (exam_id, room, ts, reported_ts, kind, reported_by, note) VALUES (?,?,?,?,?,?,?)",
                (exam_id, room, now - minutes_ago * 60.0, now, kind, actor, note))
            self._audit(actor, "missed_reported", None, f"{kind}, {minutes_ago:g} min ago")
            self._db.commit()
        return {"id": cur.lastrowid, "exam_id": exam_id, "kind": kind}

    # --------------------------------------------- data for the learning report
    def learning_rows(self, room: Optional[str] = None, since: Optional[float] = None,
                      exam_id: Optional[int] = None) -> List[Dict]:
        """Numbers only (rule, outcome, reason, who decided): no names, no seats, no video."""
        q, args = "SELECT room, kind, score, status, dismiss_reason, reviewer, reviewed_ts, created_ts, ignored_ts, " \
                  "exam_id, id FROM incidents WHERE 1=1", []
        if room:
            q += " AND room=?"; args.append(room)
        if exam_id is not None:
            q += " AND exam_id=?"; args.append(exam_id)
        if since is not None:
            q += " AND created_ts>=?"; args.append(since)
        with self._lock:
            return [dict(r) for r in self._db.execute(q, args).fetchall()]

    def missed_rows(self, room: Optional[str] = None, since: Optional[float] = None,
                    exam_id: Optional[int] = None) -> List[Dict]:
        q, args = "SELECT room, kind, ts, reported_by, exam_id FROM missed_events WHERE 1=1", []
        if room:
            q += " AND room=?"; args.append(room)
        if exam_id is not None:
            q += " AND exam_id=?"; args.append(exam_id)
        if since is not None:
            q += " AND ts>=?"; args.append(since)
        with self._lock:
            return [dict(r) for r in self._db.execute(q, args).fetchall()]

    # ------------------------------------------------------------ suggestions
    def add_suggestion(self, s: Dict) -> Dict:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO suggestions (created_ts, room, kind, setting, current_val, suggested, direction, evidence)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), s["room"], s["kind"], s["setting"], s["current"], s["suggested"], s["direction"], s["evidence"]))
            self._audit("system", "suggestion_created", None, f"{s['setting']} {s['current']} -> {s['suggested']}")
            self._db.commit()
        return self.get_suggestion(cur.lastrowid)

    def get_suggestion(self, sid: int) -> Optional[Dict]:
        with self._lock:
            r = self._db.execute("SELECT * FROM suggestions WHERE id=?", (sid,)).fetchone()
        return self._suggestion(r) if r else None

    def list_suggestions(self, status: Optional[str] = None, room: Optional[str] = None) -> List[Dict]:
        q, args = "SELECT * FROM suggestions WHERE 1=1", []
        if status:
            q += " AND status=?"; args.append(status)
        if room:
            q += " AND room=?"; args.append(room)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY id DESC", args).fetchall()
        return [self._suggestion(r) for r in rows]

    @staticmethod
    def _suggestion(r: sqlite3.Row) -> Dict:
        import json as _json
        try:
            ev = _json.loads(r["evidence"])
        except ValueError:
            ev = {}
        return {"id": r["id"], "created": r["created_ts"], "room": r["room"], "kind": r["kind"],
                "kind_label": KIND_LABELS.get(r["kind"], r["kind"]), "setting": r["setting"],
                "current": r["current_val"], "suggested": r["suggested"], "direction": r["direction"],
                "evidence": ev, "status": r["status"], "decided_by": r["decided_by"], "decided_ts": r["decided_ts"]}

    def decide_suggestion(self, sid: int, approve: bool, actor: str) -> Dict:
        actor = (actor or "").strip()
        if not actor:
            raise ReviewError("a reviewer name is required")
        with self._lock:
            r = self._db.execute("SELECT * FROM suggestions WHERE id=?", (sid,)).fetchone()
            if r is None:
                raise KeyError(sid)
            if r["status"] != "pending":
                raise ReviewError("this suggestion was already decided")
            new = "approved" if approve else "rejected"
            self._db.execute("UPDATE suggestions SET status=?, decided_by=?, decided_ts=? WHERE id=?",
                             (new, actor, time.time(), sid))
            self._audit(actor, f"suggestion_{new}", None,
                        f"{r['setting']} {r['current_val']:g} -> {r['suggested']:g}")
            self._db.commit()
        return self.get_suggestion(sid)

    # ---------------------------------------------------------------- audit
    def _audit(self, actor: str, action: str, incident_id: Optional[int] = None, detail: str = "") -> None:
        self._db.execute(
            "INSERT INTO audit (ts, actor, action, incident_id, detail) VALUES (?,?,?,?,?)",
            (time.time(), actor, action, incident_id, detail))

    def audit(self, actor: str, action: str, incident_id: Optional[int] = None, detail: str = "") -> None:
        with self._lock:
            self._audit(actor, action, incident_id, detail)
            self._db.commit()

    def list_audit(self, limit: int = 200) -> List[Dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ retention
    def purge_candidates(self, cfg: Config, now: Optional[float] = None) -> List[sqlite3.Row]:
        now = time.time() if now is None else now
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM incidents WHERE (clip_name IS NOT NULL OR thumb_name IS NOT NULL)").fetchall()
        due = []
        for r in rows:
            if r["status"] == "dismissed" and r["reviewed_ts"] and \
                    now - r["reviewed_ts"] >= cfg.retention_dismissed_hours * 3600:
                due.append(r)
            elif r["status"] == "confirmed" and r["reviewed_ts"] and \
                    now - r["reviewed_ts"] >= cfg.retention_confirmed_days * 86400:
                due.append(r)
            elif r["status"] == "pending" and now - r["created_ts"] >= cfg.retention_pending_days * 86400:
                due.append(r)
        return due

    def mark_purged(self, incident_id: int, expire: bool = False) -> None:
        with self._lock:
            if expire:
                self._db.execute(
                    "UPDATE incidents SET clip_name=NULL, thumb_name=NULL, purged_ts=?, status='expired' WHERE id=?",
                    (time.time(), incident_id))
            else:
                self._db.execute(
                    "UPDATE incidents SET clip_name=NULL, thumb_name=NULL, purged_ts=? WHERE id=?",
                    (time.time(), incident_id))
            self._audit("system", "evidence_purged", incident_id, "retention policy")
            self._db.commit()

    # --------------------------------------------------------------- export
    def export_confirmed_csv(self) -> str:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM incidents WHERE status='confirmed' ORDER BY created_ts").fetchall()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "time", "room", "student", "rule", "severity", "details",
                    "reviewed_by", "reviewed_at", "reviewer_note", "seat", "camera"])
        for r in rows:
            w.writerow([
                r["id"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_ts"])), r["room"],
                r["student_name"] or f"Student {r['track_id']}", KIND_LABELS.get(r["kind"], r["kind"]), r["severity"], r["details"],
                r["reviewer"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["reviewed_ts"])) if r["reviewed_ts"] else "",
                r["note"] or "", r["seat"] or "", r["camera"] or ""])
        return "\ufeff" + buf.getvalue()

    def close(self) -> None:
        with self._lock:
            self._db.close()
