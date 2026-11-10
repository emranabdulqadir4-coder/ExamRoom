"""Write the teacher's decision back into the roster spreadsheet (a "Mark" column).

Only a teacher's *confirmation* ever causes a write, and only for a student who has been identified.
Safety measures, because this edits the school's own file:
  * the student's row is found again by student id (or name) on every write, so sorting or editing
    the sheet after ExamGuard started cannot make it mark the wrong person
  * an ambiguous or missing row is refused, never guessed
  * a timestamped backup copy is made before the first change
  * the file is replaced atomically; if it is open in Excel (or read-only) the write fails with a
    clear message and the incident is NOT confirmed
  * clearing only removes a mark that ExamGuard itself wrote (it never erases a real grade)
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .roster import ID_COLS, NAME_COLS, Roster, Student

log = logging.getLogger("examguard.marks")


class MarkError(RuntimeError):
    pass


class MarkWriter:
    def __init__(self, roster: Roster, column: str = "mark", value: str = "Fail"):
        if roster.path is None:
            raise MarkError("the roster has no file to write to")
        self.roster, self.column, self.value = roster, column.strip().lower(), value
        self.path = Path(roster.path)
        self._lock = threading.Lock()
        self._backed_up = False

    # ------------------------------------------------------------ public
    def set(self, ref: str) -> str:
        s = self._student(ref)
        old = self._edit(s, self.value, only_if=None)
        return f"{s.name}: {self.column} = {self.value}" + (f" (was '{old}')" if old and old != self.value else "")

    def clear(self, ref: str) -> Optional[str]:
        s = self._student(ref)
        old = self._edit(s, "", only_if=self.value)
        return f"{s.name}: {self.column} cleared" if old == self.value else None

    # ----------------------------------------------------------- internal
    def _student(self, ref: str) -> Student:
        s = self.roster.get(ref)
        if s is None:
            raise MarkError("this student is not in the roster")
        return s

    def _edit(self, s: Student, new: str, only_if: Optional[str]) -> str:
        with self._lock:
            if not self.path.is_file():
                raise MarkError(f"the roster file was not found: {self.path}")
            self._backup()
            try:
                if self.path.suffix.lower() in (".xlsx", ".xlsm"):
                    return self._edit_xlsx(s, new, only_if)
                return self._edit_csv(s, new, only_if)
            except MarkError:
                raise
            except PermissionError as exc:
                raise MarkError(f"cannot write to {self.path.name}: close it in Excel (or make it "
                                f"writable) and try again") from exc
            except OSError as exc:
                raise MarkError(f"cannot write to {self.path.name}: {exc}") from exc

    def _backup(self) -> None:
        if self._backed_up:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(self.path, self.path.with_name(f"{self.path.name}.backup-{stamp}"))
        self._backed_up = True

    @staticmethod
    def _find_row(s: Student, rows: List[Tuple[int, str, str]]) -> int:
        """rows: (row number, name, student id). Prefer the id; fall back to a unique name."""
        if s.student_id:
            hits = [r for r, _, sid in rows if sid == s.student_id]
        else:
            hits = [r for r, nm, _ in rows if nm == s.name]
        if len(hits) != 1:
            raise MarkError(f"could not find exactly one row for {s.name} in the spreadsheet "
                            f"(found {len(hits)}); nothing was written")
        return hits[0]

    # -------------------------------------------------------------- xlsx
    def _edit_xlsx(self, s: Student, new: str, only_if: Optional[str]) -> str:
        from openpyxl import load_workbook
        keep_vba = self.path.suffix.lower() == ".xlsm"
        wb = load_workbook(self.path, keep_vba=keep_vba)
        ws = wb.worksheets[0]
        headers = {str(c.value).strip().lower(): c.column for c in ws[1] if c.value is not None}
        name_col = next((headers[h] for h in NAME_COLS if h in headers), None)
        id_col = next((headers[h] for h in ID_COLS if h in headers), None)
        if name_col is None:
            raise MarkError("the spreadsheet has no 'name' column")
        rows = []
        for r in range(2, ws.max_row + 1):
            nm = ws.cell(r, name_col).value
            sid = ws.cell(r, id_col).value if id_col else ""
            if nm is not None:
                rows.append((r, str(nm).strip(), "" if sid is None else _clean(sid)))
        row = self._find_row(s, rows)
        col = headers.get(self.column)
        if col is None:
            col = (max(headers.values()) if headers else 0) + 1
            ws.cell(1, col).value = self.column.capitalize()
        cell = ws.cell(row, col)
        old = "" if cell.value is None else str(cell.value).strip()
        if only_if is not None and old != only_if:
            return old
        cell.value = new if new != "" else None
        tmp = self.path.with_name(self.path.name + ".tmp")
        wb.save(tmp)
        os.replace(tmp, self.path)
        return old

    # --------------------------------------------------------------- csv
    def _edit_csv(self, s: Student, new: str, only_if: Optional[str]) -> str:
        text = self.path.read_text(encoding="utf-8-sig")
        try:
            dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        data = list(csv.reader(text.splitlines(), dialect))
        if not data:
            raise MarkError("the roster file is empty")
        headers = [h.strip().lower() for h in data[0]]
        name_i = next((headers.index(h) for h in NAME_COLS if h in headers), None)
        id_i = next((headers.index(h) for h in ID_COLS if h in headers), None)
        if name_i is None:
            raise MarkError("the spreadsheet has no 'name' column")
        rows = []
        for n, row in enumerate(data[1:], start=2):
            if len(row) > name_i and row[name_i].strip():
                sid = row[id_i].strip() if id_i is not None and len(row) > id_i else ""
                rows.append((n, row[name_i].strip(), sid))
        target = self._find_row(s, rows)
        if self.column in headers:
            col = headers.index(self.column)
        else:
            col = len(headers)
            data[0].append(self.column.capitalize())
        r = data[target - 1]
        r.extend([""] * (col + 1 - len(r)))
        old = r[col].strip()
        if only_if is not None and old != only_if:
            return old
        r[col] = new
        width = max(len(x) for x in data)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, dialect)
            for row in data:
                w.writerow(row + [""] * (width - len(row)))
        os.replace(tmp, self.path)
        return old


def _clean(v) -> str:
    """Excel stores 20230101 as a number (sometimes 20230101.0): normalise to text."""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()
