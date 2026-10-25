from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from config import DEFAULT_ACTIVE_MODEL, DEFAULT_CONFIDENCE_THRESHOLD, MAX_UPLOAD_BYTES, REQUIRED_CSV_COLUMNS, SYNTHETIC_DATA_PATH
from models.email_classifier import classify_email
from modules.action_extraction import extract_actions
from modules.preprocessing import preprocess_email
from modules.privacy import mask_personal_data


def validate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return pd.DataFrame(), [{"row": None, "error": "The uploaded dataset is empty."}]
    missing = [column for column in REQUIRED_CSV_COLUMNS if column not in df.columns]
    if missing:
        return pd.DataFrame(), [{"row": None, "error": f"Missing required columns: {', '.join(missing)}."}]
    valid_rows = []
    seen = set()
    for index, row in df.iterrows():
        row_number = int(index) + 2
        email_id = str(row.get("email_id", "")).strip()
        subject = "" if pd.isna(row.get("subject")) else str(row.get("subject", "")).strip()
        body = "" if pd.isna(row.get("body")) else str(row.get("body", "")).strip()
        timestamp = str(row.get("received_at", "")).strip()
        row_errors = []
        if not email_id:
            row_errors.append("email_id is empty")
        if not subject:
            row_errors.append("subject is empty")
        if not body:
            row_errors.append("body is empty")
        if not timestamp or pd.isna(pd.to_datetime(timestamp, errors="coerce")):
            row_errors.append("received_at is not a valid timestamp")
        if email_id in seen:
            row_errors.append("duplicate email_id within the import")
        if email_id:
            seen.add(email_id)
        if row_errors:
            errors.append({"row": row_number, "email_id": email_id, "error": "; ".join(row_errors)})
        else:
            valid_rows.append({"email_id": email_id, "subject": subject, "body": body, "received_at": pd.to_datetime(timestamp).isoformat(timespec="seconds")})
    return pd.DataFrame(valid_rows), errors


def read_csv_upload(uploaded_file) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if uploaded_file is None:
        return pd.DataFrame(), []
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        return pd.DataFrame(), [{"row": None, "error": f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB size limit."}]
    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception as exc:
        return pd.DataFrame(), [{"row": None, "error": f"Could not parse CSV: {exc}"}]
    return validate_dataframe(df)


def load_synthetic_samples() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    try:
        return validate_dataframe(pd.read_csv(SYNTHETIC_DATA_PATH))
    except Exception as exc:
        return pd.DataFrame(), [{"row": None, "error": f"Could not load the synthetic dataset: {exc}"}]


def analyze_record(record: Dict[str, Any], mask_enabled: bool = True, model_name: str = DEFAULT_ACTIVE_MODEL, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    subject = "" if record.get("subject") is None else str(record.get("subject"))
    body = "" if record.get("body") is None else str(record.get("body"))
    preprocessing = preprocess_email(subject, body)
    privacy = mask_personal_data(preprocessing["processed_text"], enabled=mask_enabled)
    processed_text = privacy["text"]
    if not processed_text:
        raise ValueError("The email has no analyzable subject or body after preprocessing.")
    classification = classify_email(processed_text, model_name=model_name, confidence_threshold=confidence_threshold)
    actions = extract_actions(processed_text, classification["urgency"])
    received_at = pd.to_datetime(record.get("received_at"), errors="coerce")
    if pd.isna(received_at):
        raise ValueError("received_at must be a valid timestamp.")
    return {
        "email_id": str(record.get("email_id") or f"manual-{datetime.now().strftime('%Y%m%d%H%M%S')}").strip(),
        "subject": subject.strip(),
        "body": body.strip(),
        "received_at": received_at.isoformat(),
        "processed_text": processed_text,
        "masked": privacy["masked"],
        "preprocessing_warnings": preprocessing["warnings"],
        "original_text": preprocessing["original_text"],
        "removed_quoted_reply": preprocessing["removed_quoted_reply"],
        "removed_signature": preprocessing["removed_signature"],
        "privacy_counts": {"emails": privacy["email_count"], "phones": privacy["phone_count"]},
        "actions": actions,
        "urgency_level": classification["urgency"]["urgency_level"],
        "urgency_evidence": classification["urgency"]["evidence"],
        "deadline_phrases": classification["urgency"]["deadline_phrases"],
        "model_name": classification["model_name"],
        "original_priority": classification["original_priority"],
        "final_priority": classification["final_priority"],
        "original_intent": classification["original_intent"],
        "final_intent": classification["final_intent"],
        "confidence": classification["confidence"],
        "intent_confidence": classification["intent_confidence"],
        "rule_changed": classification["rule_changed"],
        "rule_reason": classification["rule_reason"],
        "needs_review": classification["needs_review"],
        "review_status": "Pending review" if classification["needs_review"] else "Not required",
        "review_reasons": classification["review_reasons"],
    }


def analyze_records(records: Iterable[Dict[str, Any]], **kwargs) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    analyzed: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        try:
            analyzed.append(analyze_record(record, **kwargs))
        except Exception as exc:
            errors.append({"row": index, "email_id": record.get("email_id"), "error": str(exc)})
    return analyzed, errors
