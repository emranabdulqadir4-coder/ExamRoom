"""Optional email alerts through Gmail (SMTP).

Alerts contain text only by default (student label, rule, time and a link to the
dashboard). Attaching the thumbnail is opt-in because it sends an image of a
student through email.

Setup: turn on 2-Step Verification for the Gmail account, create an App password
(https://myaccount.google.com/apppasswords) and put it in EXAMGUARD_EMAIL_APP_PASSWORD.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from typing import Callable, Dict, List, Optional

from .config import Config
from .tls import primary_ip

log = logging.getLogger("examguard.notify")


class EmailNotifier:
    def __init__(self, cfg: Config, thumb_loader: Optional[Callable[[Dict], Optional[bytes]]] = None):
        self.user = cfg.email_user.strip()
        self.password = cfg.email_app_password.replace(" ", "")      # Google shows it with spaces
        self.recipients: List[str] = [a.strip() for a in cfg.email_to.split(",") if a.strip()] or \
                                     ([self.user] if self.user else [])
        self.host, self.port = cfg.email_smtp_host, cfg.email_smtp_port
        self.send_thumb = cfg.email_send_thumb
        host = primary_ip() if cfg.host == "0.0.0.0" else cfg.host
        self.dashboard = cfg.public_url or f"{cfg.scheme}://{host}:{cfg.port}"
        self.thumb_loader = thumb_loader

    @property
    def enabled(self) -> bool:
        return bool(self.user and self.password and self.recipients)

    def build_message(self, incident: Dict, photo: Optional[bytes] = None) -> EmailMessage:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(incident["ts"]))
        msg = EmailMessage()
        msg["Subject"] = f"Exam alert: {incident['student']} - {incident['kind_label']} ({incident['room']})"
        msg["From"] = self.user
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(
            f"Possible exam irregularity in {incident['room']}\n\n"
            f"Student:  {incident['student']}\n"
            + (f"Identity: {incident['identity_note']}\n" if incident.get("identity_note") else "") +
            f"Rule:     {incident['kind_label']} ({incident['severity']} priority)\n"
            f"Detail:   {incident['details']}\n"
            f"Time:     {when}\n\n"
            f"Review the video clip and confirm or dismiss it here:\n{self.dashboard}\n\n"
            f"This is an automated flag only, not a finding. Nothing is recorded as an "
            f"infraction until a teacher reviews it.\n")
        if photo:
            msg.add_attachment(photo, maintype="image", subtype="jpeg", filename="alert.jpg")
        return msg

    def notify(self, incident: Dict) -> None:
        """Fire and forget: never blocks or breaks the detection loop."""
        if not self.enabled:
            return
        threading.Thread(target=self._send, args=(incident,), daemon=True, name="email").start()

    def _send(self, incident: Dict) -> None:
        try:
            photo = self.thumb_loader(incident) if (self.send_thumb and self.thumb_loader) else None
            msg = self.build_message(incident, photo)
            with smtplib.SMTP_SSL(self.host, self.port, timeout=20,
                                  context=ssl.create_default_context()) as smtp:
                smtp.login(self.user, self.password)
                smtp.send_message(msg)
        except smtplib.SMTPAuthenticationError:
            log.warning("Gmail rejected the login. Use a Google App password (needs 2-Step "
                        "Verification), not your normal password.")
        except Exception as exc:                      # network problems must not stop monitoring
            log.warning("Email alert failed: %s", exc)
