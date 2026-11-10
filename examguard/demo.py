"""Camera-free demo: a simulated exam room that drives the *real* pipeline.

Six cartoon students sit at desks. The simulator plays the roles of the camera
(it renders frames), the face analyser (it reports each head's yaw) and the phone
detector. Everything after that (tracker, rules, evidence clips, encrypted
storage, dashboard, alerts) is exactly the production code.

It is clearly watermarked SIMULATION and never claims to be real detection.
"""
from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .models import Detection, FaceObs

W, H = 960, 540
R = 52                                           # head radius in pixels
CENTERS = [(220, 150), (480, 150), (740, 150), (220, 360), (480, 360), (740, 360)]
SKIN = [(150, 190, 235), (120, 160, 210), (95, 135, 190), (110, 150, 200), (170, 205, 240), (85, 120, 170)]
HAIR = [(30, 30, 35), (40, 60, 90), (25, 25, 25), (60, 90, 130), (35, 35, 50), (20, 30, 60)]
SHIRT = [(160, 90, 60), (70, 120, 90), (60, 70, 150), (120, 120, 60), (140, 80, 130), (80, 100, 160)]
BASE_YAW = [-4.0, 3.0, 24.0, -6.0, 5.0, -3.0]    # student 3 sits off-axis, like a seat at the edge of the room


BASE_PITCH = [5.0, 8.0, 3.0, 6.0, 4.0, 7.0]       # everyone looks slightly down at their paper
BASE_GAZE = [0.01, -0.02, 0.0, 0.02, -0.01, 0.01]


def _neighbour(student: int) -> int:
    """The student sitting next to this one in the same row (rows are 1-3 and 4-6)."""
    return student - 1 if student % 3 == 0 else student + 1


@dataclass
class _Action:
    kind: str            # turn | phone | down | gaze | talk | pass
    student: int         # 1-based
    start: float
    end: float
    ramp: float = 0.2
    other: int = 0       # second student for "pass"


def _env(t: float, a: _Action) -> float:
    if t < a.start or t > a.end:
        return 0.0
    return max(0.0, min(1.0, (t - a.start) / a.ramp, (a.end - t) / a.ramp))


# how long each simulated scenario lasts, in seconds (used to label the evaluation self-check)
SCENARIO_SECONDS = {"head_turn": 4.5, "glancing": 9.1, "phone": 4.5, "look_down": 5.5,
                    "eye_gaze": 3.5, "talking": 6.5, "passing": 4.5}
SCENARIOS = ("head_turn", "glancing", "phone", "look_down", "eye_gaze", "talking", "passing")


class DemoWorld:
    """Acts as video source, face analyser and object/hand detector."""
    paced = True
    name = "Simulated exam room"
    names = ["phone", "hands"]

    def __init__(self, cfg: Config, auto: bool = True, seed: int = 7):
        self.cfg = cfg
        self.auto = auto
        self.state = "simulation"
        self.n = len(CENTERS)
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._actions: List[_Action] = []
        self._period = 1.0 / cfg.process_fps
        self._due = time.time()
        self._next_auto = time.time() + 6.0
        self._fid = 0
        self._faces: List[FaceObs] = []
        self._objs: List[Detection] = []
        self._phase = [self._rng.uniform(0, 6.28) for _ in range(self.n)]

    # ------------------------------------------------------------ scenarios
    def trigger(self, kind: str, student: Optional[int] = None, delay: float = 0.0) -> dict:
        student = student or self._rng.randint(1, self.n)
        if not 1 <= student <= self.n:
            raise ValueError("student out of range")
        t0 = time.time() + delay
        with self._lock:
            if kind == "head_turn":
                self._actions.append(_Action("turn", student, t0, t0 + 4.5, 0.2))
            elif kind == "glancing":
                for i in range(5):
                    s = t0 + i * 2.0
                    self._actions.append(_Action("turn", student, s, s + 1.1, 0.15))
            elif kind == "phone":
                self._actions.append(_Action("phone", student, t0, t0 + 4.5, 0.1))
            elif kind == "look_down":
                self._actions.append(_Action("down", student, t0, t0 + 5.5, 0.3))
            elif kind == "eye_gaze":
                self._actions.append(_Action("gaze", student, t0, t0 + 3.5, 0.15))
            elif kind == "talking":
                for s in (student, _neighbour(student)):
                    self._actions.append(_Action("talk", s, t0, t0 + 6.5, 0.3))
            elif kind == "passing":
                self._actions.append(_Action("pass", student, t0, t0 + 4.5, 0.1, _neighbour(student)))
            else:
                raise ValueError("kind must be one of: " + ", ".join(SCENARIOS))
        return {"kind": kind, "student": student}

    # ----------------------------------------------------------- video source
    def read(self, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], int]:
        wait = self._due - time.time()
        if wait > 0:
            time.sleep(min(wait, timeout))
            if self._due > time.time():
                return None, -1
        self._due += self._period
        if time.time() - self._due > 1.0:
            self._due = time.time()
        now = time.time()
        if self.auto and now >= self._next_auto:
            self.trigger(self._rng.choices(SCENARIOS, [4, 2, 3, 2, 2, 2, 2])[0])
            self._next_auto = now + self._rng.uniform(14, 24)
        frame = self._render(self._state(now))
        faces, objs = self.observations_at(now)
        with self._lock:
            self._faces, self._objs = faces, objs
            self._actions = [a for a in self._actions if a.end > now - 1.0]
            self._fid += 1
            return frame, self._fid

    def observations_at(self, t: float) -> Tuple[List[FaceObs], List[Detection]]:
        """What a perfect analyser would report at time t (used by read() and by the evaluation self-check)."""
        st = self._state(t)
        faces = [FaceObs(bbox=(int(cx - R * 0.85), int(cy - R), int(R * 1.7), int(R * 2.0)), yaw=float(st["yaw"][i]),
                         pitch=float(st["pitch"][i]), gaze_x=float(st["gaze"][i]),
                         eyes_down=float(st["eyes"][i]), mouth=float(st["mouth"][i]))
                 for i, (cx, cy) in enumerate(CENTERS)]
        objs = [Detection("cell phone", round(0.78 + 0.05 * math.sin(t * 3), 2),
                          (int(CENTERS[i][0] + R * 0.8), int(CENTERS[i][1] + R * 1.35), 30, 50))
                for i in st["phones"]]
        objs += [Detection("hand", 0.9, (int(x - 18), int(y - 18), 36, 36)) for x, y in st["hands"]]
        return faces, objs

    def close(self) -> None:
        pass

    # ---------------------------------------------- analyser / detector API
    def analyze(self, frame, ts_ms: int) -> List[FaceObs]:
        with self._lock:
            return [FaceObs(bbox=f.bbox, yaw=f.yaw, pitch=f.pitch, gaze_x=f.gaze_x,
                            eyes_down=f.eyes_down, mouth=f.mouth) for f in self._faces]

    def detect(self, frame) -> List[Detection]:
        with self._lock:
            return list(self._objs)

    # ---------------------------------------------------------------- scene
    def _state(self, t: float) -> dict:
        n, ph = self.n, self._phase
        yaw = [BASE_YAW[i] + 3.0 * math.sin(0.9 * t + ph[i]) + 1.5 * math.sin(2.3 * t + 2 * ph[i]) for i in range(n)]
        pitch = [BASE_PITCH[i] + 1.5 * math.sin(0.7 * t + ph[i]) for i in range(n)]
        gaze = [BASE_GAZE[i] + 0.01 * math.sin(1.3 * t + ph[i]) for i in range(n)]
        eyes = [0.15 + 0.02 * math.sin(0.8 * t + ph[i]) for i in range(n)]
        mouth = [0.03 + 0.004 * math.sin(0.5 * t + ph[i]) for i in range(n)]
        phones, hands = set(), []
        with self._lock:
            actions = list(self._actions)
        for a in actions:
            i, e = a.student - 1, _env(t, a)
            if a.kind == "turn":
                yaw[i] += (55.0 if a.student % 2 else -55.0) * e
            elif a.kind == "phone" and a.start <= t <= a.end:
                phones.add(i)
            elif a.kind == "down":
                pitch[i] += 32.0 * e
                eyes[i] += 0.45 * e
            elif a.kind == "gaze":
                gaze[i] += (0.22 if a.student % 2 else -0.22) * e
            elif a.kind == "talk" and e > 0:
                mouth[i] = 0.03 + 0.11 * e * (0.5 + 0.5 * math.sin(2 * math.pi * 2.4 * (t - a.start) + i))
            elif a.kind == "pass" and a.start <= t <= a.end:
                (ax, ay), (bx, _) = CENTERS[i], CENTERS[a.other - 1]
                s = 1 if bx > ax else -1
                k = max(0.0, min(1.0, (t - a.start - 0.8) / 1.6))      # rest, cross, rest
                x0, x1 = ax + s * 55, bx - s * 55
                hands.append((x0 + (x1 - x0) * k, ay + R + 85))
        return {"yaw": yaw, "pitch": pitch, "gaze": gaze, "eyes": eyes, "mouth": mouth, "phones": phones, "hands": hands}

    def _render(self, st: dict) -> np.ndarray:
        img = np.empty((H, W, 3), np.uint8)
        img[:] = (196, 190, 178)                                   # wall
        img[int(H * 0.72):] = (122, 112, 100)                      # floor
        for i, (cx, cy) in enumerate(CENTERS):
            self._student(img, i, cx, cy, math.radians(st["yaw"][i]), i in st["phones"],
                          st["pitch"][i], st["gaze"][i], st["eyes"][i], st["mouth"][i])
        for x, y in st["hands"]:                                   # a hand carrying a folded note
            cv2.rectangle(img, (int(x) - 6, int(y) - 22), (int(x) + 16, int(y) - 8), (240, 240, 235), -1)
            cv2.circle(img, (int(x), int(y)), 16, (150, 185, 225), -1)
        cv2.putText(img, "SIMULATION - no real people, no real detection", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _student(img, i, cx, cy, th, phone, pitch=5.0, gaze=0.0, eyes_down=0.15, mouth=0.03) -> None:
        dy = int(max(0.0, pitch - 10.0) * 0.45)                    # head tipped down: features sit lower
        cv2.ellipse(img, (cx, cy + R + 38), (int(R * 1.35), 46), 0, 0, 360, SHIRT[i], -1)       # shoulders
        cv2.rectangle(img, (cx - 105, cy + R + 62), (cx + 105, cy + R + 112), (88, 130, 170), -1)  # desk
        cv2.rectangle(img, (cx - 105, cy + R + 62), (cx + 105, cy + R + 68), (60, 95, 130), -1)
        cv2.rectangle(img, (cx - 40, cy + R + 70), (cx + 25, cy + R + 100), (240, 240, 235), -1)   # paper
        cv2.ellipse(img, (cx, cy), (int(R * 0.82), R), 0, 0, 360, SKIN[i], -1)                    # head
        cv2.ellipse(img, (cx, cy - 6 + dy), (int(R * 0.86), int(R * 1.02)), 0, 180, 360, HAIR[i], -1)   # hair
        rh = R * 0.82
        for side in (-1, 1):                                                                       # eyes
            a = side * math.radians(26) + th
            if math.cos(a) > 0.15:
                ex, ey = int(cx + rh * 0.95 * math.sin(a)), int(cy - R * 0.08) + dy
                rx = max(2, int(7 * math.cos(a)))
                cv2.ellipse(img, (ex, ey), (rx, 5), 0, 0, 360, (250, 250, 250), -1)
                cv2.circle(img, (ex + int(2 * math.sin(th) + gaze * 24), ey + int((eyes_down - 0.15) * 7)), 2, (30, 30, 30), -1)
        nx = int(cx + rh * 1.18 * math.sin(th))                                                    # nose
        cv2.line(img, (nx, int(cy - R * 0.05) + dy), (nx, int(cy + R * 0.25) + dy), tuple(int(c * 0.7) for c in SKIN[i]), 3)
        mx, mw = int(cx + rh * 0.95 * math.sin(th)), max(3, int(14 * math.cos(th)))
        my = int(cy + R * 0.5) + dy
        if mouth > 0.06:                                                                           # open mouth
            cv2.ellipse(img, (mx, my), (mw, max(2, int(mouth * 70))), 0, 0, 360, (50, 40, 110), -1)
        else:
            cv2.line(img, (mx - mw, my), (mx + mw, my), (60, 60, 120), 2)
        if phone:
            px, py = int(cx + R * 0.8), int(cy + R * 1.35)
            cv2.rectangle(img, (px, py), (px + 30, py + 50), (35, 35, 40), -1)
            cv2.rectangle(img, (px + 3, py + 4), (px + 27, py + 44), (200, 170, 90), -1)


# ---------------------------------------------------------------------------
# Demo roster: names + cartoon "photos" for the six simulated students
DEMO_NAMES = ["Ahmed Alharbi", "Sara Alotaibi", "Omar Khan", "Layla Hassan", "Yusuf Demir", "\u0646\u0648\u0631\u0629 \u0627\u0644\u0642\u062d\u0637\u0627\u0646\u064a"]


def demo_roster():
    """Roster + fixed identities for the simulated room (no real face recognition involved)."""
    from .identify import Identity
    from .roster import Roster, Student, _small_jpeg
    world = DemoWorld(Config(), auto=False)
    frame = world._render(world._state(time.time()))
    students, mapping = [], {}
    for i, (cx, cy) in enumerate(CENTERS):
        crop = frame[max(0, cy - 70):cy + 70, max(0, cx - 62):cx + 62].copy()
        s = Student(ref=f"demo{i + 1}", name=DEMO_NAMES[i], student_id=str(20230100 + i + 1),
                    photo_bgr=crop, photo_jpeg=_small_jpeg(crop))
        students.append(s)
        mapping[i + 1] = Identity(s.ref, s.name, s.student_id, round(0.80 + 0.02 * i, 2))
    return Roster(students), mapping
