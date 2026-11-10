"""Teacher dashboard (Flask): live view, instant alerts, review queue, history, audit trail."""
from __future__ import annotations

import hmac
import io
import json
import os
import queue
import secrets
import threading
import time
from collections import defaultdict, deque
from functools import wraps
from pathlib import Path
from typing import Deque, Dict

from flask import (Flask, Response, abort, jsonify, redirect, render_template, request,
                   send_file, session, url_for)

from .. import learning, profiles
from ..healthcheck import run_checks
from ..marks import MarkError
from ..seatmap import SeatMapError, suggest_from_faces, validate_seats
from ..overlay import no_signal_jpeg
from ..service import Services, ensure_password
from ..models import DISMISS_REASONS, KIND_LABELS
from ..store import ReviewError
from ..vault import VaultError

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def create_app(svc: Services) -> Flask:
    app = Flask(__name__)
    cfg, store, vault, hub, monitor = svc.cfg, svc.store, svc.vault, svc.hub, svc.monitor
    password = ensure_password(cfg)
    app.config["DASHBOARD_PASSWORD"] = password

    secret_file = cfg.data_path / "flask_secret.key"
    if not secret_file.exists():
        secret_file.write_text(secrets.token_hex(32))
    app.secret_key = secret_file.read_text().strip()
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      SESSION_COOKIE_SECURE=cfg.https,
                      PERMANENT_SESSION_LIFETIME=12 * 3600)

    failures: Dict[str, Deque[float]] = defaultdict(deque)
    placeholder = no_signal_jpeg()

    # ----------------------------------------------------------- helpers
    def camera_names() -> list:
        return [m.camera_name or "Camera" for m in svc.all_monitors()]

    def monitor_for(arg) -> object:
        mons = svc.all_monitors()
        try:
            return mons[int(arg or 0)]
        except (ValueError, IndexError):
            abort(404)

    def reviewer() -> str:
        return session.get("reviewer", "")

    def login_required(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            if not reviewer():
                if request.path.startswith(("/api/", "/media/", "/events", "/video_feed")):
                    return jsonify(error="sign in required"), 401
                return redirect(url_for("login"))
            return fn(*a, **kw)
        return wrapper

    @app.after_request
    def headers(resp):
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith(("/api/", "/media/")):
            resp.headers["Cache-Control"] = "private, no-store"
        return resp

    # -------------------------------------------------------------- pages
    @app.get("/")
    @login_required
    def index():
        return render_template("index.html", room=cfg.room_name, reviewer=reviewer(), demo=svc.demo,
                               pre=cfg.pre_roll_seconds + cfg.post_roll_seconds, cameras=camera_names())

    @app.get("/seatmap")
    @login_required
    def seatmap_page():
        return render_template("seatmap.html", room=cfg.room_name, reviewer=reviewer(), cameras=camera_names())

    @app.get("/exam")
    @login_required
    def exam_page():
        return render_template("exam.html", room=cfg.room_name, reviewer=reviewer())

    @app.get("/health")
    @login_required
    def health_page():
        return render_template("health.html", room=cfg.room_name, reviewer=reviewer())

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = ""
        if request.method == "POST":
            ip = request.remote_addr or "?"
            now = time.time()
            q = failures[ip]
            while q and now - q[0] > 300:
                q.popleft()
            name = (request.form.get("name") or "").strip()[:60]
            if len(q) >= 5:
                error = "Too many failed attempts. Wait a few minutes."
            elif not name:
                error = "Please enter your name: it is recorded in the audit trail."
            elif hmac.compare_digest(request.form.get("password", ""), password):
                session.clear()
                session["reviewer"] = name
                session.permanent = True
                store.audit(name, "login")
                return redirect(url_for("index"))
            else:
                q.append(now)
                error = "Wrong password."
        return render_template("login.html", error=error, room=cfg.room_name)

    @app.get("/logout")
    def logout():
        if reviewer():
            store.audit(reviewer(), "logout")
        session.clear()
        return redirect(url_for("login"))

    @app.get("/healthz")
    def healthz():
        return jsonify(ok=True)

    # ---------------------------------------------------------- live video
    @app.get("/video_feed")
    @login_required
    def video_feed():
        mon = monitor_for(request.args.get("cam"))

        def gen():
            last = -1
            while True:
                jpeg, last = mon.wait_frame(last, timeout=2.0)
                jpeg = jpeg or placeholder
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                       str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    # --------------------------------------------- real-time alert channel
    @app.get("/events")
    @login_required
    def events():
        q = hub.subscribe()

        def gen():
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        kind, payload = q.get(timeout=15)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield f"event: {kind}\ndata: {payload}\n\n"
            finally:
                hub.unsubscribe(q)

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ----------------------------------------------------------------- API
    @app.get("/api/status")
    @login_required
    def api_status():
        s = monitor.status()
        s.update(pending=store.count("pending"), demo=svc.demo, room=cfg.room_name,
                 clients=hub.clients, email=svc.notifier.enabled, role=svc.role,
                 edge=svc.link.info() if svc.link is not None else None,
                 roster=len(svc.roster) if svc.roster else 0, profile=cfg.profile,
                 cameras=[m.status() for m in svc.all_monitors()])
        return jsonify(s)

    @app.get("/api/incidents")
    @login_required
    def api_incidents():
        status = request.args.get("status") or None
        limit = max(1, min(int(request.args.get("limit", 200)), 500))
        return jsonify(incidents=store.list(status, limit), pending=store.count("pending"))

    @app.get("/api/incidents/<int:iid>")
    @login_required
    def api_incident(iid):
        inc = store.get(iid)
        return (jsonify(inc) if inc else (jsonify(error="not found"), 404))

    @app.post("/api/incidents/<int:iid>/review")
    @login_required
    def api_review(iid):
        data = request.get_json(silent=True) or {}
        decision = str(data.get("decision", ""))
        reason = str(data.get("reason") or "") or None
        prev = store.get(iid)
        if prev is None:
            return jsonify(error="not found"), 404
        ref, mark_note, mark_warning = prev["student_ref"], "", ""
        if reason is not None and (decision != "dismiss" or reason not in DISMISS_REASONS):
            return jsonify(error="unknown dismiss reason" if decision == "dismiss" else "only a dismissal has a reason"), 400
        if decision not in ("confirm", "dismiss") or prev["status"] == "expired":
            try:                                     # let the store produce the proper error message
                store.review(iid, decision, reviewer(), "")
            except ReviewError as exc:
                return jsonify(error=str(exc)), 409
        if decision == "confirm" and ref and svc.marks:
            try:                                     # write first: if the file is locked, nothing is confirmed
                mark_note = svc.marks.set(ref)
            except MarkError as exc:
                return jsonify(error=str(exc)), 409
            store.audit(reviewer(), "mark_written", iid, mark_note)
        try:
            inc = store.review(iid, decision, reviewer(), str(data.get("note", "")), reason)
        except ReviewError as exc:
            if mark_note and store.count_confirmed_for(ref) == 0:      # undo the mark we just wrote
                try:
                    svc.marks.clear(ref)
                except MarkError:
                    pass
            return jsonify(error=str(exc)), 409
        if decision == "dismiss" and prev["status"] == "confirmed" and ref and svc.marks \
                and store.count_confirmed_for(ref) == 0:
            try:                                     # a confirmation was withdrawn: remove our mark
                msg = svc.marks.clear(ref)
                if msg:
                    store.audit(reviewer(), "mark_cleared", iid, msg)
                    mark_note = msg
            except MarkError as exc:
                mark_warning = f"The incident was dismissed, but the spreadsheet mark could not be removed: {exc}"
        hub.publish("incident_updated", inc)
        return jsonify({**inc, "mark_note": mark_note, "mark_warning": mark_warning})

    # ------------------------------------- ignore, reasons, exams, missed
    @app.get("/api/reasons")
    @login_required
    def api_reasons():
        return jsonify(reasons=[{"key": k, "label": v} for k, v in DISMISS_REASONS.items()],
                       kinds=[{"key": k, "label": v} for k, v in KIND_LABELS.items()] +
                             [{"key": "other", "label": "Something else"}])

    @app.post("/api/incidents/<int:iid>/ignore")
    @login_required
    def api_ignore(iid):
        on = bool((request.get_json(silent=True) or {}).get("on", True))
        try:
            inc = store.ignore(iid, reviewer(), on)
        except KeyError:
            return jsonify(error="not found"), 404
        except ReviewError as exc:
            return jsonify(error=str(exc)), 409
        hub.publish("incident_updated", inc)
        return jsonify(inc)

    @app.post("/api/incidents/<int:iid>/reason")
    @login_required
    def api_reason(iid):
        try:
            inc = store.set_dismiss_reason(iid, str((request.get_json(silent=True) or {}).get("reason", "")), reviewer())
        except KeyError:
            return jsonify(error="not found"), 404
        except ReviewError as exc:
            return jsonify(error=str(exc)), 409
        hub.publish("incident_updated", inc)
        return jsonify(inc)

    @app.get("/api/exam")
    @login_required
    def api_exam():
        return jsonify(current=store.current_exam(cfg.room_name), exams=store.list_exams(cfg.room_name))

    @app.post("/api/exam/start")
    @login_required
    def api_exam_start():
        try:
            return jsonify(store.start_exam(cfg.room_name, reviewer()))
        except ReviewError as exc:
            return jsonify(error=str(exc)), 409

    @app.post("/api/exam/finish")
    @login_required
    def api_exam_finish():
        try:
            return jsonify(store.finish_exam(cfg.room_name, reviewer()))
        except ReviewError as exc:
            return jsonify(error=str(exc)), 409

    @app.get("/api/exams/<int:eid>")
    @login_required
    def api_exam_summary(eid):
        out = store.exam_summary(eid)
        return (jsonify(out) if out else (jsonify(error="not found"), 404))

    @app.post("/api/missed")
    @login_required
    def api_missed():
        data = request.get_json(silent=True) or {}
        try:
            out = store.add_missed(cfg.room_name, str(data.get("kind", "")), reviewer(),
                                   str(data.get("note", "")), data.get("minutes_ago", 0))
        except ReviewError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(out)

    # -------------------------------------------- learning report + suggestions
    @app.get("/api/learning/report")
    @login_required
    def api_learning_report():
        eid = request.args.get("exam")
        return jsonify(learning.build_report(store, room=cfg.room_name,
                                             exam_id=int(eid) if eid and eid.isdigit() else None))

    @app.get("/api/learning/report.csv")
    @login_required
    def api_learning_csv():
        store.audit(reviewer(), "export_learning_report")
        return Response(learning.report_csv(learning.build_report(store, room=cfg.room_name)), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=field-report.csv"})

    @app.get("/api/learning/suggestions")
    @login_required
    def api_suggestions():
        return jsonify(suggestions=store.list_suggestions(room=cfg.room_name),
                       can_approve=learning.can_approve(cfg, reviewer()),
                       overrides=learning.read_overrides(cfg.data_path),
                       thresholds={"min_decided": learning.MIN_DECIDED})

    @app.post("/api/learning/suggestions/refresh")
    @login_required
    def api_suggestions_refresh():
        made = learning.propose(store, cfg)
        store.audit(reviewer(), "suggestions_refreshed", None, f"{len(made)} new")
        return jsonify(new=len(made), suggestions=store.list_suggestions(room=cfg.room_name))

    @app.post("/api/learning/suggestions/<int:sid>/<verb>")
    @login_required
    def api_suggestion_decide(sid, verb):
        if verb not in ("approve", "reject"):
            abort(404)
        try:
            fn = learning.approve if verb == "approve" else learning.reject
            out = fn(store, cfg, sid, reviewer())
        except KeyError:
            return jsonify(error="not found"), 404
        except ReviewError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify({**out, "restart_needed": verb == "approve"})

    @app.get("/api/audit")
    @login_required
    def api_audit():
        return jsonify(audit=store.list_audit(min(int(request.args.get("limit", 200)), 1000)))

    @app.get("/api/export.csv")
    @login_required
    def api_export():
        store.audit(reviewer(), "export_confirmed_csv")
        return Response(store.export_confirmed_csv(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=confirmed-incidents.csv"})

    @app.post("/api/demo/trigger")
    @login_required
    def api_demo_trigger():
        if not svc.demo:
            return jsonify(error="demo controls are only available in demo mode"), 404
        data = request.get_json(silent=True) or {}
        try:
            student = int(data["student"]) if data.get("student") else None
            worlds = svc.worlds or [svc.world]
            world = worlds[int(data.get("cam", 0))]
            return jsonify(world.trigger(str(data.get("kind", "")), student))
        except (ValueError, TypeError, IndexError) as exc:
            return jsonify(error=str(exc)), 400

    # --------------------------------------------------------------- roster
    @app.get("/api/roster")
    @login_required
    def api_roster():
        return jsonify(roster=svc.roster.public_list() if svc.roster else [],
                       write_marks=bool(svc.marks), mark_column=cfg.marks_column.capitalize(),
                       mark_value=cfg.mark_value)

    @app.get("/api/roster/<ref>/photo")
    @login_required
    def api_roster_photo(ref):
        s = svc.roster.get(ref) if svc.roster else None
        if s is None or s.photo_jpeg is None:
            abort(404)
        return send_file(io.BytesIO(s.photo_jpeg), mimetype="image/jpeg", max_age=300)

    @app.post("/api/incidents/<int:iid>/student")
    @login_required
    def api_set_student(iid):
        """Teacher corrects who the flagged student is (empty ref = unidentified)."""
        ref = str((request.get_json(silent=True) or {}).get("ref") or "")
        student = svc.roster.get(ref) if ref else None
        if ref and student is None:
            return jsonify(error="unknown student"), 400
        prev = store.get(iid)
        if prev is None:
            return jsonify(error="not found"), 404
        old, new = prev["student_ref"], (student.ref if student else None)
        marks_active = bool(svc.marks and prev["status"] == "confirmed" and old != new)
        mark_note, mark_warning = "", ""
        if marks_active and new:
            try:
                mark_note = svc.marks.set(new)
            except MarkError as exc:
                return jsonify(error=str(exc)), 409
            store.audit(reviewer(), "mark_written", iid, mark_note)
        inc = store.set_student(iid, new, student.name if student else None, reviewer())
        if marks_active and old and store.count_confirmed_for(old) == 0:
            try:
                msg = svc.marks.clear(old)
                if msg:
                    store.audit(reviewer(), "mark_cleared", iid, msg)
            except MarkError as exc:
                mark_warning = f"The earlier student's mark could not be removed: {exc}"
        hub.publish("incident_updated", inc)
        return jsonify({**inc, "mark_note": mark_note, "mark_warning": mark_warning})

    # ------------------------------------------------------------ seat map
    @app.get("/api/seatmap")
    @login_required
    def api_seatmap():
        return jsonify(seats=[x.to_dict() for x in svc.seatmap.seats()], cameras=camera_names(),
                       roster=svc.roster.public_list() if svc.roster else [])

    @app.put("/api/seatmap")
    @login_required
    def api_seatmap_save():
        data = request.get_json(silent=True) or {}
        try:
            seats = validate_seats(data.get("seats", []), n_cameras=len(svc.all_monitors()))
        except SeatMapError as exc:
            return jsonify(error=str(exc)), 400
        if svc.roster is not None:
            for x in seats:
                if x.student_ref and svc.roster.get(x.student_ref) is None:
                    return jsonify(error=f"seat '{x.label}': unknown student"), 400
            refs = [x.student_ref for x in seats if x.student_ref]
            dupes = {r for r in refs if refs.count(r) > 1}
            if dupes:
                names = ", ".join(svc.roster.get(r).name for r in sorted(dupes))
                return jsonify(error=f"the same student is assigned to two seats: {names}"), 400
        else:
            for x in seats:
                x.student_ref = ""
        svc.seatmap.replace(seats)
        store.audit(reviewer(), "seatmap_saved", None, f"{len(seats)} seats")
        note = ""
        if svc.role == "server" and svc.link is not None:
            sent = svc.link.publish_seatmap(seats)        # the edge box does the seat matching, so it needs the map
            note = ("Sent to the edge box." if sent else
                    "Saved here, but the edge box could not be reached just now: it will be sent again automatically.")
            store.audit(reviewer(), "seatmap_sent_to_edge" if sent else "seatmap_send_failed")
        return jsonify(seats=[x.to_dict() for x in seats], note=note)

    @app.get("/api/seatmap/snapshot")
    @login_required
    def api_seatmap_snapshot():
        jpeg = monitor_for(request.args.get("cam")).snapshot_jpeg()
        if jpeg is None:
            abort(404)
        return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/seatmap/suggest")
    @login_required
    def api_seatmap_suggest():
        mon = monitor_for(request.args.get("cam"))
        seats = suggest_from_faces(mon.snapshot_faces(), mon.camera_index)
        return jsonify(seats=[x.to_dict() for x in seats])

    # ------------------------------------------------------- system + profile
    @app.get("/api/health")
    @login_required
    def api_health():
        return jsonify(run_checks(svc))

    @app.get("/api/profile")
    @login_required
    def api_profile():
        saved = profiles.read_saved_profile(cfg.data_path)
        return jsonify(current=cfg.profile, saved=saved, profiles=profiles.catalogue(),
                       supervised=bool(os.environ.get("EXAMGUARD_SUPERVISED")),
                       pinned_by_env=bool(os.environ.get("EXAMGUARD_PROFILE")))

    @app.post("/api/profile")
    @login_required
    def api_profile_set():
        name = str((request.get_json(silent=True) or {}).get("profile", ""))
        if name not in profiles.PROFILES:
            return jsonify(error="unknown profile"), 400
        profiles.save_profile(cfg.data_path, name)
        store.audit(reviewer(), "profile_chosen", None, name)
        note = ""
        if os.environ.get("EXAMGUARD_PROFILE"):
            note = ("EXAMGUARD_PROFILE is set in your environment or .env file and overrides this choice: "
                    "remove it there.")
        return jsonify(saved=name, current=cfg.profile, restart_needed=name != cfg.profile, note=note)

    @app.post("/api/restart")
    @login_required
    def api_restart():
        """Restart ExamGuard so a new profile applies. Only possible when started with start.py (it relaunches us)."""
        if not os.environ.get("EXAMGUARD_SUPERVISED"):
            return jsonify(error="Restart ExamGuard yourself (Ctrl+C, then start it again). "
                                 "Starting it with start.py makes this button work."), 409
        store.audit(reviewer(), "restart_requested")

        def later():
            time.sleep(1.0)
            try:
                svc.stop()
            finally:
                os._exit(75)
        threading.Thread(target=later, daemon=True).start()
        return jsonify(ok=True)

    # ------------------------------------------------------------- evidence
    @app.get("/media/<int:iid>/<which>")
    @login_required
    def media(iid, which):
        if which not in ("clip", "thumb"):
            abort(404)
        names = store.media_names(iid)
        name = names and names[which]
        if not name:
            abort(404)
        try:
            data = vault.get(name)
        except VaultError:
            abort(404)
        if which == "clip":
            rng = request.range
            if rng is None or not rng.ranges or rng.ranges[0][0] == 0:      # log each viewing once
                store.audit(reviewer(), "view_clip", iid)
        mimetype = "video/mp4" if which == "clip" else "image/jpeg"
        return send_file(io.BytesIO(data), mimetype=mimetype, conditional=True, max_age=0)

    return app
