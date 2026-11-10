"""Evaluation tool and the stronger (staged) face detector, tested with stand-ins for the real models."""
import csv

import cv2
import numpy as np
import pytest

from examguard import evaluation as ev
from examguard import vision
from examguard.config import Config
from examguard.headpose import EYE_OUTER_A, EYE_OUTER_B, NOSE_TIP
from examguard.models import Event, FaceObs


def E(ts, kind, tid=1):
    return Event(ts=ts, track_id=tid, kind=kind, severity="medium", score=0.5, details="x")


# ------------------------------------------------------------------ evaluation
def test_parse_time_accepts_seconds_and_minutes():
    assert ev.parse_time("42") == 42 and ev.parse_time("1:05") == 65 and ev.parse_time("1:01:01") == 3661


def test_load_labels_and_clear_errors(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("start,end,kind\n0:10,0:15,head_turn\n1:00,1:04,Phone\n", encoding="utf-8")
    labs = ev.load_labels(str(p))
    assert [(l.start, l.end, l.kind) for l in labs] == [(10, 15, "head_turn"), (60, 64, "phone")]
    p.write_text("start,end,kind\n0:10,0:15,cheating\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown kind"):
        ev.load_labels(str(p))
    p.write_text("start,end,kind\n20,10,phone\n", encoding="utf-8")
    with pytest.raises(ValueError, match="before start"):
        ev.load_labels(str(p))
    p.write_text("start,end,kind\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no label rows"):
        ev.load_labels(str(p))


def test_score_recall_precision_false_alarms_and_delay():
    labels = [ev.Label(10, 15, "head_turn"), ev.Label(40, 45, "head_turn"), ev.Label(70, 75, "phone")]
    events = [E(12, "head_turn"), E(13, "repeated_glancing"),      # two alerts for one real event: both correct
              E(100, "head_turn"),                                  # nothing real there: false alarm
              E(72, "phone")]
    rep = ev.score(events, labels, duration=1800)
    ht, ph = rep.groups["head_turn"], rep.groups["phone"]
    assert (ht.labels, ht.detected, ht.alerts, ht.correct_alerts) == (2, 1, 3, 2)
    assert ht.recall == 0.5 and ht.precision == pytest.approx(2 / 3)
    assert [m.start for m in ht.missed] == [40] and ht.false_alarm_times == [100]
    assert ht.delays == [2]
    assert ph.recall == 1.0 and ph.precision == 1.0
    assert rep.false_alarms == 1 and rep.false_alarms_per_hour == pytest.approx(2.0)


def test_score_tolerance_and_kind_filter():
    labels = [ev.Label(10, 12, "talking")]
    assert ev.score([E(14.5, "talking")], labels, 60, tolerance=3).groups["talking"].detected == 1
    assert ev.score([E(16.0, "talking")], labels, 60, tolerance=3).groups["talking"].detected == 0
    rep = ev.score([E(5, "phone"), E(11, "talking")], labels, 60, kinds=["talking"])
    assert list(rep.groups) == ["talking"] and rep.false_alarms == 0       # the phone alert is not judged


def test_report_text_warns_about_few_labels_and_events_csv(tmp_path):
    rep = ev.score([E(12, "phone")], [ev.Label(10, 14, "phone")], 600)
    text = ev.format_report(rep)
    assert "Fewer than 20 labelled events" in text and "False alarms: 0" in text
    out = tmp_path / "a.csv"
    ev.events_to_csv([E(75, "phone"), E(5, "head_turn")], str(out))
    rows = list(csv.reader(open(out, encoding="utf-8-sig")))
    assert rows[0][:3] == ["time", "seconds", "kind"] and rows[1][0] == "0:05" and rows[2][0] == "1:15"


def test_demo_self_check_catches_every_scripted_event_without_false_alarms():
    cfg = Config(phone_enabled=True)
    events, duration, labels = ev.demo_run(cfg)
    rep = ev.score(events, labels, duration)
    for kind, g in rep.groups.items():
        if g.labels:
            assert g.recall == 1.0, (kind, ev.format_report(rep))
    assert rep.false_alarms == 0, ev.format_report(rep)


def test_video_steps_uses_video_time_and_resizes(tmp_path):
    path = str(tmp_path / "v.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    for _ in range(50):                                          # 5 s at 10 fps
        vw.write(np.zeros((240, 320, 3), np.uint8))
    vw.release()
    seen, widths = [], []

    class Stub:
        def analyze(self, frame, ts_ms):
            seen.append(ts_ms)
            widths.append(frame.shape[1])
            return [FaceObs(bbox=(10, 10, 50, 50), yaw=0.0)]
    cfg = Config(process_fps=5.0, max_width=160)
    steps = list(ev.video_steps(path, cfg, Stub()))
    ts = [s[0] for s in steps]
    assert ts[0] == 0 and ts[1] == pytest.approx(0.2, abs=0.01) and 24 <= len(ts) <= 26
    assert set(widths) == {160}                                  # frames are shrunk to max_width like the live monitor
    assert seen == sorted(set(seen))                             # strictly increasing millisecond stamps
    assert list(ev.video_steps(path, cfg, Stub(), max_seconds=1.0))[-1][0] <= 1.0
    with pytest.raises(FileNotFoundError):
        list(ev.video_steps(str(tmp_path / "missing.mp4"), cfg, Stub()))


# ------------------------------------------------------------------ staged detector
def test_tiles_cover_the_frame_and_overlap():
    assert vision.tile_rects(1000, 600, 1) == [(0, 0, 1000, 600)]
    tiles = vision.tile_rects(1000, 600, 2)
    assert len(tiles) == 4 and min(t[0] for t in tiles) == 0 and max(t[2] for t in tiles) == 1000
    assert tiles[0][2] > 500 > tiles[1][0]                       # neighbouring tiles overlap


def test_merge_boxes_removes_duplicates_from_overlapping_tiles_but_keeps_neighbours():
    boxes = [(100, 100, 40, 40, 0.9), (102, 101, 40, 40, 0.7), (150, 100, 40, 40, 0.8)]
    kept = vision.merge_boxes(boxes)
    assert len(kept) == 2 and kept[0][4] == 0.9


def test_crop_maps_back_to_frame_pixels():
    frame = np.zeros((400, 600, 3), np.uint8)
    crop, x0, y0, scale = vision.crop_for_face(frame, (300, 200, 30, 30), min_side=192)
    assert crop.shape[0] >= 190 and scale > 1                    # a 30 px face is enlarged for the landmark model
    cx = x0 + (crop.shape[1] / 2) / scale
    assert abs(cx - 315) <= 2                                    # the crop centre maps back onto the face
    crop2, a, b, s2 = vision.crop_for_face(frame, (590, 390, 10, 10))   # face at the frame edge stays inside
    assert 0 <= a and a + crop2.shape[1] / s2 <= 600 + 1 and 0 <= b and b + crop2.shape[0] / s2 <= 400 + 1


def fake_points(cx, cy):
    pts = [(cx + (i % 7), cy + (i // 70)) for i in range(478)]
    pts[NOSE_TIP], pts[EYE_OUTER_A], pts[EYE_OUTER_B] = (cx, cy + 10), (cx - 30, cy - 10), (cx + 30, cy - 10)
    return pts


class FakeDetector:
    name = "Fake"

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, frame):
        return list(self.boxes)


def test_staged_analyzer_reads_each_face_in_frame_coordinates():
    frame = np.zeros((400, 600, 3), np.uint8)
    seen = []

    def landmarks(crop):
        seen.append(crop.shape)
        h, w = crop.shape[:2]
        return fake_points(w / 2, h / 2), None, {}
    cfg = Config(max_students=10)
    an = vision.StagedFaceAnalyzer(cfg, FakeDetector([(400, 200, 30, 30, 0.9), (100, 100, 30, 30, 0.8)]), landmarks)
    faces = an.analyze(frame, 0)
    assert len(faces) == 2 and all(f.yaw is not None for f in faces)
    assert abs(faces[0].center[0] - 115) < 15 and abs(faces[1].center[0] - 415) < 15      # centred on each face
    assert all(s[0] >= 190 for s in seen)


def test_staged_analyzer_keeps_unreadable_faces_and_caps_the_count():
    frame = np.zeros((400, 600, 3), np.uint8)
    an = vision.StagedFaceAnalyzer(Config(max_students=2),
                                   FakeDetector([(10 + 60 * i, 50, 30, 30, 0.9 - i * 0.1) for i in range(5)]),
                                   lambda crop: None)
    faces = an.analyze(frame, 0)
    assert len(faces) == 2 and all(f.yaw is None and f.bbox[2] == 30 for f in faces)


class Small:
    name = "primary"

    def __init__(self, width):
        self.width, self.closed = width, False

    def analyze(self, frame, ts_ms):
        return [FaceObs(bbox=(i * 80, 0, self.width, self.width), yaw=0.0) for i in range(4)]

    def close(self):
        self.closed = True


def test_auto_switches_once_when_faces_stay_small():
    cfg = Config(min_face_px=40)
    staged = Small(60)
    an = vision.AutoFaceAnalyzer(cfg, Small(20), lambda: staged, switch_after_frames=5)
    for _ in range(4):
        an.analyze(None, 0)
    assert not an.switched
    an.analyze(None, 0)
    assert an.switched and an.analyze(None, 0)[0].bbox[2] == 60


def test_auto_stays_put_when_faces_are_big_or_the_stronger_detector_is_missing():
    cfg = Config(min_face_px=40)
    big = vision.AutoFaceAnalyzer(cfg, Small(80), lambda: 1 / 0, switch_after_frames=3)
    for _ in range(10):
        big.analyze(None, 0)
    assert not big.switched

    def boom():
        raise RuntimeError("model could not be downloaded")
    an = vision.AutoFaceAnalyzer(cfg, Small(20), boom, switch_after_frames=3)
    for _ in range(10):                                            # never raises: monitoring continues with MediaPipe
        assert an.analyze(None, 0)
    assert not an.switched


def test_auto_counter_resets_if_faces_become_big_again():
    cfg = Config(min_face_px=40)
    p = Small(20)
    an = vision.AutoFaceAnalyzer(cfg, p, lambda: Small(60), switch_after_frames=5)
    for _ in range(4):
        an.analyze(None, 0)
    p.width = 80
    an.analyze(None, 0)
    p.width = 20
    for _ in range(4):
        an.analyze(None, 0)
    assert not an.switched


def test_unknown_detector_mode_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="auto, mediapipe or staged"):
        vision.load_face_analyzer(Config(face_detector="retina"))
