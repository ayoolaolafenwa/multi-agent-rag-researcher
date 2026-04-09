import sqlite3
from pathlib import Path

from .helpers import build_evidence_context

"""
Memory
=====================================================================================
Handles short term memory for the multi-agent workflow.

The memory keeps the latest user query for a session and the latest cached evidence
retrieved for that session. This allows the orchestrator to support followup questions
and reuse evidence when it is still relevant.
"""

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
MEMORY_DB_PATH = UTILS_DIR / "memory.db"


# get sqlite connection for memory operations
def get_memory_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# initialize memory tables for session query and cached evidence
def init_memory() -> None:
    with get_memory_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_memory (
                session_id TEXT PRIMARY KEY,
                last_user_query TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_memory (
                session_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


# save latest user query for a given session
def save_last_user_query(session_id: str, user_message: str) -> None:
    with get_memory_connection() as conn:
        conn.execute(
            """
            INSERT INTO session_memory (session_id, last_user_query)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_user_query = excluded.last_user_query,
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, user_message),
        )


# save latest retrieved evidence for a given session
def save_evidence(session_id: str, query: str, evidence_json: str) -> None:
    with get_memory_connection() as conn:
        conn.execute(
            """
            INSERT INTO evidence_memory (session_id, query, evidence_json)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                query = excluded.query,
                evidence_json = excluded.evidence_json,
                created_at = CURRENT_TIMESTAMP
            """,
            (session_id, query, evidence_json),
        )


# get latest session query and cached evidence context
def get_session_context(session_id: str) -> dict[str, str]:
    with get_memory_connection() as conn:
        session_row = conn.execute(
            """
            SELECT last_user_query
            FROM session_memory
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        evidence_row = conn.execute(
            """
            SELECT query, evidence_json
            FROM evidence_memory
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

    context = {
        "last_user_query": session_row["last_user_query"] if session_row else "",
        "cached_query": "",
        "cached_evidence_json": "",
        "cached_evidence_summary": "None",
    }
    if not evidence_row:
        return context

    evidence_json = evidence_row["evidence_json"]
    evidence_context = build_evidence_context(evidence_json)
    if not evidence_context["has_evidence"]:
        return context

    return {
        **context,
        "cached_query": evidence_row["query"],
        "cached_evidence_json": evidence_json,
        "cached_evidence_summary": evidence_context["summary"],
    }