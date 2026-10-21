import json
import os
import sqlite3
import threading
import time
from pathlib import Path


STATE_DB_PATH = Path(
    os.getenv("AI_AGENT_STATE_DB", "data/agent_state.sqlite3")
).resolve()
STATE_LOCK = threading.RLock()
CACHE_TTL_SECONDS = 600
JOB_TTL_SECONDS = 86_400
CONVERSATION_TTL_SECONDS = 30 * 86_400


def _connect() -> sqlite3.Connection:
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STATE_DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_cache (
            dataset_id TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            value TEXT NOT NULL,
            expires_at REAL NOT NULL,
            PRIMARY KEY (dataset_id, cache_key)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_jobs (
            job_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            sheet TEXT,
            question TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            status TEXT NOT NULL,
            result TEXT,
            error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_threads (
            thread_id TEXT PRIMARY KEY,
            updated_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            parent_run_id TEXT,
            graph_name TEXT NOT NULL,
            event TEXT NOT NULL,
            component TEXT NOT NULL,
            duration_ms INTEGER,
            created_at REAL NOT NULL
        )
        """
    )
    return connection


def get_cached_analysis(dataset_id: str, cache_key: str) -> dict | None:
    with STATE_LOCK:
        connection = _connect()
        try:
            row = connection.execute(
                """
                SELECT value FROM analysis_cache
                WHERE dataset_id = ? AND cache_key = ? AND expires_at > ?
                """,
                (dataset_id, cache_key, time.time()),
            ).fetchone()
            if row is None:
                connection.execute(
                    "DELETE FROM analysis_cache WHERE expires_at <= ?",
                    (time.time(),),
                )
                connection.commit()
                return None
            return json.loads(row["value"])
        finally:
            connection.close()


def cache_analysis(dataset_id: str, cache_key: str, result: dict) -> None:
    with STATE_LOCK:
        connection = _connect()
        try:
            now = time.time()
            connection.execute(
                """
                INSERT OR REPLACE INTO analysis_cache
                    (dataset_id, cache_key, value, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    cache_key,
                    json.dumps(result, ensure_ascii=False, allow_nan=False),
                    now + CACHE_TTL_SECONDS,
                ),
            )
            connection.execute(
                "DELETE FROM analysis_cache WHERE expires_at <= ?", (now,)
            )
            connection.commit()
        finally:
            connection.close()


def create_analysis_job(
    job_id: str,
    dataset_id: str,
    sheet: str | None,
    question: str,
    conversation_id: str,
) -> None:
    now = time.time()
    with STATE_LOCK:
        connection = _connect()
        try:
            active_job = connection.execute(
                """
                SELECT 1 FROM analysis_jobs
                WHERE dataset_id = ? AND conversation_id = ?
                  AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (dataset_id, conversation_id),
            ).fetchone()
            if active_job is not None:
                raise ValueError("Hội thoại này đang có tác vụ phân tích chưa hoàn tất.")
            connection.execute(
                """
                INSERT INTO analysis_jobs
                    (job_id, dataset_id, sheet, question, conversation_id, status,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    job_id,
                    dataset_id,
                    sheet,
                    question,
                    conversation_id,
                    now,
                    now,
                ),
            )
            connection.commit()
        finally:
            connection.close()


def update_analysis_job(
    job_id: str,
    status: str,
    *,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    if status not in {"running", "succeeded", "failed"}:
        raise ValueError(f"Trạng thái job không hợp lệ: {status}.")
    with STATE_LOCK:
        connection = _connect()
        try:
            connection.execute(
                """
                UPDATE analysis_jobs
                SET status = ?, result = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    json.dumps(result, ensure_ascii=False, allow_nan=False)
                    if result is not None
                    else None,
                    error,
                    time.time(),
                    job_id,
                ),
            )
            connection.commit()
        finally:
            connection.close()


def get_analysis_job(job_id: str) -> dict | None:
    with STATE_LOCK:
        connection = _connect()
        try:
            now = time.time()
            connection.execute(
                "DELETE FROM analysis_jobs WHERE updated_at <= ?",
                (now - JOB_TTL_SECONDS,),
            )
            connection.commit()
            row = connection.execute(
                """
                SELECT job_id, status, result, error, conversation_id
                FROM analysis_jobs WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "job_id": row["job_id"],
                "status": row["status"],
                "result": json.loads(row["result"]) if row["result"] else None,
                "error": row["error"],
                "conversation_id": row["conversation_id"],
            }
        finally:
            connection.close()


def mark_interrupted_jobs() -> None:
    with STATE_LOCK:
        connection = _connect()
        try:
            now = time.time()
            connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'failed',
                    error = 'API đã khởi động lại trước khi tác vụ hoàn tất.',
                    updated_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (now,),
            )
            connection.execute(
                "DELETE FROM analysis_jobs WHERE updated_at <= ?",
                (now - JOB_TTL_SECONDS,),
            )
            connection.execute(
                "DELETE FROM analysis_cache WHERE expires_at <= ?", (now,)
            )
            connection.commit()
        finally:
            connection.close()


def touch_conversation(thread_id: str) -> None:
    with STATE_LOCK:
        connection = _connect()
        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO conversation_threads (thread_id, updated_at)
                VALUES (?, ?)
                """,
                (thread_id, time.time()),
            )
            connection.commit()
        finally:
            connection.close()


def conversation_exists(thread_id: str) -> bool:
    with STATE_LOCK:
        connection = _connect()
        try:
            return connection.execute(
                "SELECT 1 FROM conversation_threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone() is not None
        finally:
            connection.close()


def get_expired_conversations() -> list[str]:
    with STATE_LOCK:
        connection = _connect()
        try:
            rows = connection.execute(
                "SELECT thread_id FROM conversation_threads WHERE updated_at <= ?",
                (time.time() - CONVERSATION_TTL_SECONDS,),
            ).fetchall()
            return [row["thread_id"] for row in rows]
        finally:
            connection.close()


def delete_conversation_record(thread_id: str) -> None:
    with STATE_LOCK:
        connection = _connect()
        try:
            connection.execute(
                "DELETE FROM conversation_threads WHERE thread_id = ?", (thread_id,)
            )
            connection.commit()
        finally:
            connection.close()


def get_dataset_conversations(dataset_id: str) -> list[str]:
    with STATE_LOCK:
        connection = _connect()
        try:
            rows = connection.execute(
                "SELECT thread_id FROM conversation_threads WHERE thread_id LIKE ?",
                (f"{dataset_id}:%",),
            ).fetchall()
            return [row["thread_id"] for row in rows]
        finally:
            connection.close()


def delete_dataset_records(dataset_id: str) -> None:
    with STATE_LOCK:
        connection = _connect()
        try:
            connection.execute(
                "DELETE FROM analysis_cache WHERE dataset_id = ?", (dataset_id,)
            )
            connection.execute(
                "DELETE FROM analysis_jobs WHERE dataset_id = ?", (dataset_id,)
            )
            connection.commit()
        finally:
            connection.close()


def record_trace_event(
    trace_id: str,
    run_id: str,
    parent_run_id: str | None,
    graph_name: str,
    event: str,
    component: str,
    duration_ms: int | None,
) -> None:
    with STATE_LOCK:
        connection = _connect()
        try:
            now = time.time()
            connection.execute(
                """
                INSERT INTO agent_traces
                    (trace_id, run_id, parent_run_id, graph_name, event,
                     component, duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    run_id,
                    parent_run_id,
                    graph_name,
                    event,
                    component,
                    duration_ms,
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM agent_traces WHERE created_at <= ?",
                (now - 14 * 86_400,),
            )
            connection.commit()
        finally:
            connection.close()


def list_trace_events(limit: int = 100) -> list[dict]:
    with STATE_LOCK:
        connection = _connect()
        try:
            rows = connection.execute(
                """
                SELECT trace_id, run_id, parent_run_id, graph_name, event,
                       component, duration_ms, created_at
                FROM agent_traces
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()
