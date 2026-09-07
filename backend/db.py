"""
SQLite persistence layer.

Kept intentionally thin — raw SQL with sqlite3 so the data model is
completely transparent. No ORM magic hiding what's happening underneath.
"""

import sqlite3
import json
from pathlib import Path

DB_PATH = Path("facts.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # concurrent reads during writes
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                page_count  INTEGER,
                fact_count  INTEGER DEFAULT 0,
                quality_score REAL,
                uploaded_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS facts (
                id                TEXT PRIMARY KEY,
                document_id       TEXT NOT NULL,
                claim             TEXT NOT NULL,
                fact_type         TEXT NOT NULL,
                temporal_scope    TEXT,
                entity_scope      TEXT,
                exact_quote       TEXT NOT NULL,
                page_number       INTEGER,
                confidence        REAL NOT NULL,
                uncertainty_reason TEXT,
                embedding         TEXT,
                created_at        TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );

            CREATE TABLE IF NOT EXISTS relationships (
                id                     TEXT PRIMARY KEY,
                fact_a_id              TEXT NOT NULL,
                fact_b_id              TEXT NOT NULL,
                relationship           TEXT NOT NULL,
                reasoning              TEXT NOT NULL,
                reconciliation_context TEXT,
                confidence             REAL NOT NULL,
                created_at             TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (fact_a_id) REFERENCES facts(id),
                FOREIGN KEY (fact_b_id) REFERENCES facts(id)
            );
        """)


# ── Documents ──────────────────────────────────────────────────────────────

def insert_document(doc_id: str, filename: str, page_count: int, quality_score: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents (id, filename, page_count, quality_score) VALUES (?, ?, ?, ?)",
            (doc_id, filename, page_count, quality_score),
        )


def update_document_fact_count(doc_id: str, count: int):
    with get_conn() as conn:
        conn.execute("UPDATE documents SET fact_count = ? WHERE id = ?", (count, doc_id))


def list_documents() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None


def delete_document(doc_id: str):
    """Cascade delete: remove the document, its facts, and any relationships involving those facts."""
    with get_conn() as conn:
        fact_ids = [
            r["id"] for r in
            conn.execute("SELECT id FROM facts WHERE document_id = ?", (doc_id,)).fetchall()
        ]
        if fact_ids:
            placeholders = ",".join("?" * len(fact_ids))
            conn.execute(
                f"DELETE FROM relationships WHERE fact_a_id IN ({placeholders}) OR fact_b_id IN ({placeholders})",
                fact_ids + fact_ids,
            )
        conn.execute("DELETE FROM facts WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


# ── Facts ──────────────────────────────────────────────────────────────────

def insert_fact(fact: dict):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO facts
               (id, document_id, claim, fact_type, temporal_scope, entity_scope,
                exact_quote, page_number, confidence, uncertainty_reason, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact["id"], fact["document_id"], fact["claim"], fact["fact_type"],
                fact.get("temporal_scope"), fact.get("entity_scope"),
                fact["exact_quote"], fact["page_number"],
                fact["confidence"], fact.get("uncertainty_reason"),
                fact.get("embedding"),
            ),
        )


def get_facts_for_document(doc_id: str) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM facts WHERE document_id = ? ORDER BY page_number, confidence DESC",
            (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_facts_except_document(doc_id: str) -> list:
    """Fetch facts from all OTHER documents — used when running cross-doc comparison for a new upload."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM facts WHERE document_id != ?", (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_facts() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM facts ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


# ── Relationships ──────────────────────────────────────────────────────────

def insert_relationship(rel: dict):
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO relationships
               (id, fact_a_id, fact_b_id, relationship, reasoning, reconciliation_context, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                rel["id"], rel["fact_a_id"], rel["fact_b_id"],
                rel["relationship"], rel["reasoning"],
                rel.get("reconciliation_context"), rel["confidence"],
            ),
        )


def get_all_relationships() -> list:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                r.*,
                fa.claim        AS claim_a,
                fa.fact_type    AS type_a,
                fa.temporal_scope AS temporal_a,
                fa.entity_scope   AS scope_a,
                fa.exact_quote  AS quote_a,
                fa.page_number  AS page_a,
                fa.confidence   AS conf_a,
                fa.document_id  AS doc_id_a,
                fb.claim        AS claim_b,
                fb.fact_type    AS type_b,
                fb.temporal_scope AS temporal_b,
                fb.entity_scope   AS scope_b,
                fb.exact_quote  AS quote_b,
                fb.page_number  AS page_b,
                fb.confidence   AS conf_b,
                fb.document_id  AS doc_id_b,
                da.filename     AS filename_a,
                db.filename     AS filename_b
            FROM relationships r
            JOIN facts fa ON r.fact_a_id = fa.id
            JOIN facts fb ON r.fact_b_id = fb.id
            JOIN documents da ON fa.document_id = da.id
            JOIN documents db ON fb.document_id = db.id
            ORDER BY r.created_at DESC
        """).fetchall()
        return [dict(row) for row in rows]


def relationship_exists(fact_a_id: str, fact_b_id: str) -> bool:
    """Avoid computing the same pair twice (in either direction)."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT 1 FROM relationships
               WHERE (fact_a_id = ? AND fact_b_id = ?)
                  OR (fact_a_id = ? AND fact_b_id = ?)""",
            (fact_a_id, fact_b_id, fact_b_id, fact_a_id),
        ).fetchone()
        return row is not None
