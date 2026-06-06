"""
Data lineage tracking and AI-generated SQL over cohort metadata.

Lineage: record where each artifact came from and which step produced it.
Text-to-SQL: the planning agent can answer a natural-language question about the
synthetic cohort by generating SQL against a SQLite metadata table. The generated
query is returned alongside the result so it can be logged and reviewed — an
auditable form of AI-generated SQL.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic

_SQL_SYSTEM_TMPL = """You translate a natural-language question into a single, safe,
read-only SQLite SELECT statement over a table named `cohort`.
The table has exactly these columns: {columns}.
Return ONLY the SQL, no prose, no code fences.
Never write INSERT, UPDATE, DELETE, DROP, or ALTER."""


class DataLineage:
    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def record(self, artifact: str, source: str, produced_by: str) -> None:
        self._records.append(
            {"artifact": artifact, "source": source, "produced_by": produced_by, "ts": time.time()}
        )

    def history(self) -> List[Dict[str, Any]]:
        return list(self._records)


class CohortSQL:
    """AI-generated, read-only SQL over the synthetic cohort metadata table."""

    def __init__(self, sqlite_path: str, client: Optional[Anthropic] = None,
                 model: Optional[str] = None):
        self.sqlite_path = sqlite_path
        self.client = client or Anthropic()
        self.model = model or os.environ.get("IMEMORY_EXECUTOR_MODEL", "claude-sonnet-4-20250514")

    def _get_columns(self) -> str:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(cohort)").fetchall()]
        finally:
            conn.close()
        return ", ".join(cols)

    def _generate_sql(self, question: str) -> str:
        columns = self._get_columns()
        system  = _SQL_SYSTEM_TMPL.format(columns=columns)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": question}],
        )
        sql = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not sql.lower().lstrip().startswith("select"):
            raise ValueError(f"Refusing non-SELECT statement: {sql!r}")
        return sql

    def ask(self, question: str) -> Tuple[str, List[tuple]]:
        """Return (generated_sql, rows). The SQL is surfaced for audit."""
        sql = self._generate_sql(question)
        conn = sqlite3.connect(self.sqlite_path)
        try:
            rows = conn.execute(sql).fetchall()
        finally:
            conn.close()
        return sql, rows
