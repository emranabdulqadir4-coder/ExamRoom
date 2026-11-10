"""Student roster: a spreadsheet (.xlsx) or CSV with each student's name and photo.

Columns (header names are case-insensitive, extra columns are ignored):

    name          required   e.g. "Ahmed Alharbi"  (Arabic names are fine, save CSV as UTF-8)
    student_id    optional   e.g. "20231045"
    photo         optional   file name of the photo, looked up in the photos folder
                             (default: a "photos" folder next to the roster, else the same folder)

In an .xlsx file the photo may instead be **inserted into the sheet** (Insert > Picture) on the
student's row. The photo is only used to suggest who a face belongs to; a teacher always confirms.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

log = logging.getLogger("examguard.roster")

NAME_COLS = ("name", "full name", "full_name", "student name", "student_name", "student")
ID_COLS = ("student_id", "student id", "id", "number")
PHOTO_COLS = ("photo", "image", "picture", "photo_file", "photo file")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RosterError(RuntimeError):
    pass


@dataclass
class Student:
    ref: str                        # unique key (student id, else row number)
    name: str
    student_id: str = ""
    photo_bgr: Optional[np.ndarray] = field(default=None, repr=False)
    photo_jpeg: Optional[bytes] = field(default=None, repr=False)   # small copy shown in the dashboard
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class Roster:
    def __init__(self, students: List[Student], warnings: Optional[List[str]] = None,
                 path: Optional[str] = None):
        self.path = path                       # the spreadsheet file (needed to write marks back)
        self.students = students
        self.warnings = warnings or []
        self._by_ref: Dict[str, Student] = {s.ref: s for s in students}

    def __len__(self) -> int:
        return len(self.students)

    def get(self, ref: Optional[str]) -> Optional[Student]:
        return self._by_ref.get(ref or "")

    def public_list(self) -> List[Dict]:
        return [{"ref": s.ref, "name": s.name, "student_id": s.student_id,
                 "has_photo": s.photo_jpeg is not None} for s in self.students]

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str, photos_dir: str = "") -> "Roster":
        p = Path(path)
        if not p.is_file():
            raise RosterError(f"roster file not found: {p}")
        folders = [Path(photos_dir)] if photos_dir else [p.parent / "photos", p.parent]
        if p.suffix.lower() in (".xlsx", ".xlsm"):
            rows, embedded = _read_xlsx(p)
        elif p.suffix.lower() in (".csv", ".txt"):
            rows, embedded = _read_csv(p), {}
        else:
            raise RosterError("roster must be .xlsx or .csv")

        students: List[Student] = []
        warnings: List[str] = []
        seen = set()
        for n, row in rows:
            name = _pick(row, NAME_COLS)
            if not name:
                continue
            sid = _normalize_id(_pick(row, ID_COLS))
            ref = _slug(sid or f"row{n}")
            if ref in seen:
                ref = f"{ref}-{n}"
            seen.add(ref)
            raw = embedded.get(n)
            if raw is None:
                fname = _pick(row, PHOTO_COLS)
                if fname:
                    raw = _read_photo(fname, folders)
            img = _decode(raw) if raw else None
            if img is None:
                warnings.append(f"{name}: no usable photo (face matching skipped for this student)")
            s = Student(ref=ref, name=name, student_id=sid)
            if img is not None:
                s.photo_bgr, s.photo_jpeg = img, _small_jpeg(img)
            students.append(s)
        if not students:
            raise RosterError("the roster has no students (is there a 'name' column?)")
        return cls(students, warnings, str(p))


# --------------------------------------------------------------------- helpers
def _normalize_id(v: str) -> str:
    """Excel often hands back 20230101.0 for a numeric id."""
    return v[:-2] if re.fullmatch(r"\d+\.0", v) else v


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip()).strip("-") or "student"


def _pick(row: Dict[str, str], names) -> str:
    for n in names:
        v = row.get(n)
        if v:
            return v.strip()
    return ""


def _norm_row(headers: List[str], values) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for h, v in zip(headers, values):
        if h and v is not None:
            out[h] = str(v).strip()
    return out


def _read_csv(p: Path):
    text = p.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(text.splitlines(), dialect)
    rows = list(reader)
    if not rows:
        raise RosterError("the roster file is empty")
    headers = [h.strip().lower() for h in rows[0]]
    return [(i + 2, _norm_row(headers, r)) for i, r in enumerate(rows[1:])]


def _read_xlsx(p: Path):
    from openpyxl import load_workbook
    wb = load_workbook(p, data_only=True)
    ws = wb.worksheets[0]
    data = list(ws.iter_rows(values_only=True))
    if not data:
        raise RosterError("the roster sheet is empty")
    headers = [str(h).strip().lower() if h is not None else "" for h in data[0]]
    rows = [(i + 2, _norm_row(headers, r)) for i, r in enumerate(data[1:])]
    embedded: Dict[int, bytes] = {}
    for img in getattr(ws, "_images", []):               # pictures inserted into the sheet
        try:
            row = img.anchor._from.row + 1               # 0-based -> 1-based sheet row
            embedded[row] = img._data()
        except Exception:
            continue
    return rows, embedded


def _read_photo(fname: str, folders: List[Path]) -> Optional[bytes]:
    base = Path(fname).name                               # never follow ../ out of the photos folder
    if Path(base).suffix.lower() not in IMAGE_EXT:
        return None
    for folder in folders:
        f = folder / base
        if f.is_file():
            return f.read_bytes()
    return None


def _decode(raw: bytes) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None or min(img.shape[:2]) < 40:
        return None
    return img


def _small_jpeg(img: np.ndarray, side: int = 240) -> bytes:
    h, w = img.shape[:2]
    scale = side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
