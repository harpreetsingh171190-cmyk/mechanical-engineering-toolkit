"""Small SQLite history store for engineering calculations and sensor readings."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path


class HistoryStore:
    """Append-only calculation history backed by a local SQLite file.

    Every operation opens and *closes* its own connection (``sqlite3``'s own
    context manager only commits, it does not close), so the database file is
    never left locked - important on Windows.
    """

    def __init__(self, path: str | Path = "mechanical_history.sqlite3") -> None:
        self.path = str(path)
        with closing(self._connect()) as con, con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS records ("
                "id INTEGER PRIMARY KEY, created_at REAL NOT NULL, "
                "kind TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(self, kind: str, payload: dict) -> int:
        if not kind.strip():
            raise ValueError("kind must not be empty")
        encoded = json.dumps(payload, allow_nan=False)
        with closing(self._connect()) as con, con:
            cur = con.execute(
                "INSERT INTO records(created_at, kind, payload) VALUES(?, ?, ?)",
                (time.time(), kind, encoded),
            )
            return int(cur.lastrowid)

    def list(self, kind: str | None = None, limit: int = 100) -> list[dict]:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        query = "SELECT id, created_at, kind, payload FROM records"
        args: tuple = ()
        if kind:
            query += " WHERE kind = ?"
            args = (kind,)
        query += " ORDER BY id DESC LIMIT ?"
        with closing(self._connect()) as con:
            rows = con.execute(query, (*args, limit)).fetchall()
        return [{"id": r[0], "created_at": r[1], "kind": r[2], "payload": json.loads(r[3])} for r in rows]
