"""
governance/audit_store.py
─────────────────────────
Durable, tamper-evident audit trail (compliance / auditability requirement).

Every governed inference, governance action, human approval decision, booking, support
conversation turn and outcome feedback is appended to a SQLite table. Each row stores the
SHA-256 of (previous row hash + canonical JSON of this row), so any edit or deletion breaks the
chain and is caught by :meth:`AuditStore.verify`. The store survives process restarts, which also
lets the supervisor restore its action log and pending approvals after a crash.

Personal data is masked before it is written (see :func:`redact`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.getenv("RB_STATE_DIR", ROOT / "runtime"))
DB_PATH = STATE_DIR / "audit.db"
GENESIS = "0" * 64

# ── Personal-data masking ─────────────────────────────────────────────────────
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
_AADHAAR = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")
_SENSITIVE_KEYS = {"passenger_name", "name", "email", "phone", "mobile", "aadhaar", "address", "card_number"}


def _mask_text(s: str) -> str:
    s = _EMAIL.sub("[email]", s)
    s = _AADHAAR.sub("[id-number]", s)
    s = _CARD.sub("[card]", s)
    s = _PHONE.sub("[phone]", s)
    return s


def redact(obj: Any) -> Any:
    """Recursively mask personal data (names, e-mails, phone / ID / card numbers)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in _SENSITIVE_KEYS and v not in (None, ""):
                out[k] = "[redacted]"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return _mask_text(obj)
    return obj


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))


class AuditStore:
    def __init__(self, path: Path = DB_PATH) -> None:
        self._lock = threading.Lock()
        self.path = Path(path)
        self._ok = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS audit ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL,"
                " module TEXT, ref TEXT, actor TEXT, payload TEXT NOT NULL,"
                " prev_hash TEXT NOT NULL, hash TEXT NOT NULL)"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_audit_kind ON audit(kind, id)")
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_audit_ref ON audit(ref)")
            self._db.commit()
            row = self._db.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            self._last_hash = row[0] if row else GENESIS
        except Exception:
            # Never take the platform down because the disk is read-only: fall back to memory.
            self._db = sqlite3.connect(":memory:", check_same_thread=False)
            self._db.execute(
                "CREATE TABLE audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, module TEXT,"
                " ref TEXT, actor TEXT, payload TEXT, prev_hash TEXT, hash TEXT)"
            )
            self._last_hash = GENESIS
            self._ok = False

    @property
    def durable(self) -> bool:
        return self._ok

    def append(self, kind: str, payload: Dict[str, Any], module: str = "", ref: str = "",
               actor: str = "system") -> str:
        clean = redact(payload)
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            body = _canonical({"ts": ts, "kind": kind, "module": module, "ref": ref, "actor": actor, "payload": clean})
            h = hashlib.sha256((self._last_hash + body).encode()).hexdigest()
            try:
                self._db.execute(
                    "INSERT INTO audit (ts, kind, module, ref, actor, payload, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?)",
                    (ts, kind, module, ref, actor, _canonical(clean), self._last_hash, h),
                )
                self._db.commit()
                self._last_hash = h
            except Exception:
                return ""
            return h

    def query(self, kind: Optional[str] = None, module: Optional[str] = None, ref: Optional[str] = None,
              limit: int = 100) -> List[dict]:
        sql, args = "SELECT id, ts, kind, module, ref, actor, payload, hash FROM audit WHERE 1=1", []
        if kind:
            sql += " AND kind = ?"; args.append(kind)
        if module:
            sql += " AND module = ?"; args.append(module)
        if ref:
            sql += " AND ref = ?"; args.append(ref)
        sql += " ORDER BY id DESC LIMIT ?"; args.append(int(limit))
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [{"id": r[0], "ts": r[1], "kind": r[2], "module": r[3], "ref": r[4], "actor": r[5],
                 "payload": json.loads(r[6]), "hash": r[7]} for r in rows]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT kind, COUNT(*) FROM audit GROUP BY kind").fetchall()
        return {k: n for k, n in rows}

    def verify(self) -> dict:
        """Recompute the hash chain; report the first tampered row, if any."""
        prev = GENESIS
        n = 0
        with self._lock:
            rows = self._db.execute(
                "SELECT id, ts, kind, module, ref, actor, payload, prev_hash, hash FROM audit ORDER BY id"
            ).fetchall()
        for r in rows:
            n += 1
            body = _canonical({"ts": r[1], "kind": r[2], "module": r[3], "ref": r[4], "actor": r[5],
                               "payload": json.loads(r[6])})
            expect = hashlib.sha256((prev + body).encode()).hexdigest()
            if r[7] != prev or r[8] != expect:
                return {"intact": False, "entries_checked": n, "first_broken_id": r[0], "durable": self._ok}
            prev = r[8]
        return {"intact": True, "entries_checked": n, "head_hash": prev, "durable": self._ok,
                "path": str(self.path) if self._ok else ":memory:"}


audit_store = AuditStore()
