from __future__ import annotations

from datetime import datetime

import pandas as pd

from config import DB_PATH
from modules.action_extraction import extract_actions
from modules.database import delete_all, get_email, init_db, update_correction, upsert_email
from modules.email_service import analyze_record, validate_dataframe
from modules.preprocessing import preprocess_email
from modules.privacy import mask_personal_data
from modules.urgency_detection import detect_urgency


def test_preprocessing_removes_quote_and_normalizes_whitespace():
    result = preprocess_email(" Hello   team ", "Please review.\n\n> old reply\nRegards,\nA")
    assert result["processed_text"].startswith("Hello team")
    assert result["removed_quoted_reply"] is True
    assert result["removed_signature"] is True


def test_privacy_masking():
    result = mask_personal_data("Contact me at person@example.com or +1 (555) 123-4567")
    assert result["masked"] is True
    assert "[EMAIL]" in result["text"]
    assert "[PHONE]" in result["text"]


def test_urgency_and_deadline_detection():
    result = detect_urgency("Production system is down. Please respond by tomorrow.")
    assert result["urgency_level"] in {"High", "Medium"}
    assert result["deadline_phrases"]
    not_urgent = detect_urgency("This is not urgent and no response is needed.")
    assert not not_urgent["evidence"]


def test_action_extraction():
    urgency = detect_urgency("Please approve the request before 5 PM today.")
    actions = extract_actions("Please approve the request before 5 PM today.", urgency)
    assert actions
    assert actions[0]["status"] == "Pending"


def test_invalid_csv_rows_are_reported():
    df = pd.DataFrame([
        {"email_id": "ok", "subject": "Hello", "body": "Body text", "received_at": "2026-10-17 12:00"},
        {"email_id": "bad", "subject": "", "body": "", "received_at": "not-a-date"},
    ])
    valid, errors = validate_dataframe(df)
    assert len(valid) == 1
    assert len(errors) == 1


def test_model_analysis_and_low_confidence_flag():
    record = analyze_record({"email_id": "test-model", "subject": "Production API is down", "body": "Please investigate immediately and respond by tomorrow.", "received_at": datetime.now().isoformat()})
    assert record["final_priority"] in {"Critical", "High", "Medium", "Low"}
    assert record["final_intent"]
    assert 0 <= record["confidence"] <= 1
    assert record["urgency_level"] in {"High", "Medium", "Low"}


def test_database_insert_retrieve_and_correction(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    init_db(db_path)
    record = analyze_record({"email_id": "db-1", "subject": "Routine update", "body": "This is a routine notification for your information.", "received_at": datetime.now().isoformat()})
    upsert_email(record, db_path)
    assert get_email("db-1", db_path)["email_id"] == "db-1"
    update_correction("db-1", "Low", "Information", "Human confirmed routine.", db_path)
    corrected = get_email("db-1", db_path)
    assert corrected["user_priority"] == "Low"
    assert corrected["correction_notes"] == "Human confirmed routine."


def test_export_shape():
    record = analyze_record({"email_id": "export-1", "subject": "Please send details", "body": "Please send the details before Friday.", "received_at": datetime.now().isoformat()})
    assert "body" in record
    assert "actions" in record
