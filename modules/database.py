import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from config import DB_PATH


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path | str = DB_PATH) -> None:
    connection = get_connection(db_path)
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id TEXT NOT NULL UNIQUE,
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            received_at TEXT NOT NULL,
            processed_text TEXT NOT NULL,
            masked INTEGER NOT NULL DEFAULT 0,
            model_name TEXT NOT NULL,
            original_priority TEXT,
            final_priority TEXT,
            original_intent TEXT,
            final_intent TEXT,
            confidence REAL,
            intent_confidence REAL,
            urgency_level TEXT,
            urgency_evidence TEXT,
            deadline_phrases TEXT,
            actions TEXT,
            rule_changed INTEGER NOT NULL DEFAULT 0,
            rule_reason TEXT,
            needs_review INTEGER NOT NULL DEFAULT 0,
            review_status TEXT NOT NULL DEFAULT 'Not required',
            user_priority TEXT,
            user_intent TEXT,
            correction_notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            reviewed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id TEXT NOT NULL,
            original_priority TEXT,
            original_intent TEXT,
            confidence REAL,
            final_priority TEXT,
            user_priority TEXT,
            user_intent TEXT,
            review_status TEXT,
            correction_notes TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.commit()
    connection.close()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def _decode(row: sqlite3.Row) -> Dict[str, Any]:
    item = dict(row)
    for field in ("urgency_evidence", "deadline_phrases", "actions"):
        try:
            item[field] = json.loads(item[field] or "[]")
        except (TypeError, json.JSONDecodeError):
            item[field] = []
    item["masked"] = bool(item.get("masked"))
    item["rule_changed"] = bool(item.get("rule_changed"))
    item["needs_review"] = bool(item.get("needs_review"))
    return item


def upsert_email(record: Dict[str, Any], db_path: Path | str = DB_PATH) -> None:
    init_db(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    connection = get_connection(db_path)
    connection.execute(
        """
        INSERT INTO emails (
            email_id, subject, body, received_at, processed_text, masked, model_name,
            original_priority, final_priority, original_intent, final_intent, confidence,
            intent_confidence, urgency_level, urgency_evidence, deadline_phrases, actions,
            rule_changed, rule_reason, needs_review, review_status, user_priority,
            user_intent, correction_notes, created_at, updated_at, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(email_id) DO UPDATE SET
            subject=excluded.subject, body=excluded.body, received_at=excluded.received_at,
            processed_text=excluded.processed_text, masked=excluded.masked, model_name=excluded.model_name,
            original_priority=excluded.original_priority, final_priority=excluded.final_priority,
            original_intent=excluded.original_intent, final_intent=excluded.final_intent,
            confidence=excluded.confidence, intent_confidence=excluded.intent_confidence,
            urgency_level=excluded.urgency_level, urgency_evidence=excluded.urgency_evidence,
            deadline_phrases=excluded.deadline_phrases, actions=excluded.actions,
            rule_changed=excluded.rule_changed, rule_reason=excluded.rule_reason,
            needs_review=excluded.needs_review, review_status=excluded.review_status,
            updated_at=excluded.updated_at
        """,
        (
            record.get("email_id"), record.get("subject", ""), record.get("body", ""), record.get("received_at"),
            record.get("processed_text", ""), int(bool(record.get("masked"))), record.get("model_name", ""),
            record.get("original_priority"), record.get("final_priority"), record.get("original_intent"),
            record.get("final_intent"), record.get("confidence"), record.get("intent_confidence"),
            record.get("urgency_level"), _json(record.get("urgency_evidence", [])), _json(record.get("deadline_phrases", [])),
            _json(record.get("actions", [])), int(bool(record.get("rule_changed"))), record.get("rule_reason", ""),
            int(bool(record.get("needs_review"))), record.get("review_status", "Not required"), record.get("user_priority"),
            record.get("user_intent"), record.get("correction_notes", ""), record.get("created_at", now), now, record.get("reviewed_at"),
        ),
    )
    connection.commit()
    connection.close()


def get_email(email_id: str, db_path: Path | str = DB_PATH) -> Dict[str, Any] | None:
    init_db(db_path)
    connection = get_connection(db_path)
    row = connection.execute("SELECT * FROM emails WHERE email_id = ?", (email_id,)).fetchone()
    connection.close()
    return _decode(row) if row else None


def list_emails(db_path: Path | str = DB_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    connection = get_connection(db_path)
    rows = connection.execute("SELECT * FROM emails ORDER BY datetime(received_at) DESC, id DESC").fetchall()
    connection.close()
    return [_decode(row) for row in rows]


def update_correction(email_id: str, priority: str | None, intent: str | None, notes: str = "", db_path: Path | str = DB_PATH) -> None:
    record = get_email(email_id, db_path)
    if not record:
        raise ValueError("Email record was not found.")
    now = datetime.now().isoformat(timespec="seconds")
    review_status = "Reviewed"
    connection = get_connection(db_path)
    connection.execute(
        "UPDATE emails SET user_priority=?, user_intent=?, final_priority=?, final_intent=?, correction_notes=?, review_status=?, needs_review=0, reviewed_at=?, updated_at=? WHERE email_id=?",
        (priority, intent, priority or record.get("final_priority"), intent or record.get("final_intent"), notes, review_status, now, now, email_id),
    )
    connection.execute(
        "INSERT INTO feedback_history (email_id, original_priority, original_intent, confidence, final_priority, user_priority, user_intent, review_status, correction_notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (email_id, record.get("original_priority"), record.get("original_intent"), record.get("confidence"), record.get("final_priority"), priority, intent, review_status, notes, now),
    )
    connection.commit()
    connection.close()


def update_actions(email_id: str, actions: Iterable[Dict[str, Any]], db_path: Path | str = DB_PATH) -> None:
    connection = get_connection(db_path)
    connection.execute("UPDATE emails SET actions=?, updated_at=? WHERE email_id=?", (_json(list(actions)), datetime.now().isoformat(timespec="seconds"), email_id))
    connection.commit()
    connection.close()


def mark_reviewed(email_id: str, reviewed: bool = True, db_path: Path | str = DB_PATH) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    connection = get_connection(db_path)
    connection.execute("UPDATE emails SET review_status=?, needs_review=?, reviewed_at=?, updated_at=? WHERE email_id=?", ("Reviewed" if reviewed else "Pending review", 0 if reviewed else 1, now if reviewed else None, now, email_id))
    connection.commit()
    connection.close()


def feedback_records(db_path: Path | str = DB_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    connection = get_connection(db_path)
    rows = connection.execute("SELECT * FROM feedback_history ORDER BY datetime(created_at) DESC, id DESC").fetchall()
    connection.close()
    return [dict(row) for row in rows]


def counts(db_path: Path | str = DB_PATH) -> Dict[str, int]:
    rows = list_emails(db_path)
    return {
        "total": len(rows),
        "critical": sum(row.get("final_priority") == "Critical" for row in rows),
        "high": sum(row.get("final_priority") == "High" for row in rows),
        "medium": sum(row.get("final_priority") == "Medium" for row in rows),
        "low": sum(row.get("final_priority") == "Low" for row in rows),
        "review": sum(row.get("review_status") == "Pending review" or row.get("needs_review") for row in rows),
        "deadlines": sum(bool(row.get("deadline_phrases")) for row in rows),
        "actions": sum(any(action.get("status") == "Pending" for action in row.get("actions", [])) for row in rows),
    }


def delete_all(db_path: Path | str = DB_PATH) -> None:
    connection = get_connection(db_path)
    connection.execute("DELETE FROM emails")
    connection.execute("DELETE FROM feedback_history")
    connection.commit()
    connection.close()


def records_dataframe(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    rows = list_emails(db_path)
    if not rows:
        return pd.DataFrame()
    compact = []
    for row in rows:
        item = dict(row)
        item["action_item_count"] = len(item.get("actions", []))
        item["deadline_count"] = len(item.get("deadline_phrases", []))
        item.pop("body", None)
        item.pop("processed_text", None)
        compact.append(item)
    return pd.DataFrame(compact)
