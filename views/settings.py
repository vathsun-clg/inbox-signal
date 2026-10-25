from __future__ import annotations

import pandas as pd
import streamlit as st

from config import DEFAULT_ACTIVE_MODEL, DEFAULT_CONFIDENCE_THRESHOLD, REQUIRED_CSV_COLUMNS, SYNTHETIC_DATA_PATH
from modules.database import counts, delete_all, feedback_records, list_emails
from modules.email_service import analyze_records, load_synthetic_samples, read_csv_upload
from modules.database import upsert_email
from views.common import page_header


def _export_frame(records, include_body: bool = False) -> pd.DataFrame:
    rows = []
    for record in records:
        item = {
            "email_id": record.get("email_id"), "subject": record.get("subject"), "received_at": record.get("received_at"),
            "predicted_priority": record.get("original_priority"), "suggested_priority": record.get("final_priority"),
            "predicted_intent": record.get("original_intent"), "confidence": record.get("confidence"),
            "urgency_level": record.get("urgency_level"), "deadline_count": len(record.get("deadline_phrases", [])),
            "action_item_count": len(record.get("actions", [])), "review_status": record.get("review_status"),
        }
        if include_body:
            item["body"] = record.get("body")
        rows.append(item)
    return pd.DataFrame(rows)


def render() -> None:
    page_header("Controls", "Settings & data management", "Import approved samples, export focused reports, and keep the local store under your control.")
    summary = counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Stored emails", summary["total"])
    c2.metric("Feedback rows", len(feedback_records()))
    c3.metric("SQLite", "ready" if SYNTHETIC_DATA_PATH.parent.exists() else "unavailable")
    st.markdown("#### Import approved CSV samples")
    st.caption(f"Required columns: {', '.join(REQUIRED_CSV_COLUMNS)}. Invalid rows are reported and are not silently discarded.")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    import_mode = st.radio("Import source", ["Uploaded file", "Built-in synthetic dataset"], horizontal=True)
    if st.button("Validate and import", type="primary"):
        df, errors = read_csv_upload(upload) if import_mode == "Uploaded file" else load_synthetic_samples()
        if errors:
            st.warning(f"{len(errors)} invalid row(s) or import issue(s) detected.")
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
        if not df.empty:
            with st.spinner("Analyzing and saving valid records…"):
                analyzed, analysis_errors = analyze_records(df.to_dict("records"), mask_enabled=True, model_name=st.session_state.get("active_model", DEFAULT_ACTIVE_MODEL), confidence_threshold=st.session_state.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
                for record in analyzed:
                    upsert_email(record)
            st.success(f"Imported {len(analyzed)} valid record(s). Existing email IDs were safely updated rather than duplicated.")
            if analysis_errors:
                st.warning("Some valid-looking rows could not be analyzed.")
                st.dataframe(pd.DataFrame(analysis_errors), use_container_width=True, hide_index=True)
    st.download_button("Download sample CSV", SYNTHETIC_DATA_PATH.read_bytes(), "sample_emails.csv", "text/csv")
    st.divider()
    st.markdown("#### Export focused reports")
    records = list_emails()
    report = st.selectbox("Report", ["Triage summary", "Priority list", "Manual-review queue", "Confirmed action items", "Feedback history"])
    include_body = st.checkbox("Include complete email bodies", value=False, help="Off by default to reduce accidental exposure in summary exports.")
    if report == "Feedback history":
        export_df = pd.DataFrame(feedback_records())
    else:
        selected = records
        if report == "Priority list":
            selected = [r for r in records if r.get("final_priority") in {"Critical", "High"}]
        elif report == "Manual-review queue":
            selected = [r for r in records if r.get("review_status") == "Pending review" or r.get("needs_review")]
        elif report == "Confirmed action items":
            selected = [r for r in records if any(a.get("status") == "Accepted" for a in r.get("actions", []))]
        export_df = _export_frame(selected, include_body)
    st.download_button("Download selected report", export_df.to_csv(index=False).encode("utf-8"), "email_triage_report.csv", "text/csv", disabled=export_df.empty)
    st.divider()
    st.markdown("#### Local model controls")
    st.session_state["active_model"] = st.selectbox("Active prediction model", ["logistic_regression", "multinomial_nb"], index=0 if st.session_state.get("active_model", DEFAULT_ACTIVE_MODEL) == "logistic_regression" else 1)
    st.session_state["confidence_threshold"] = st.slider("Manual-review threshold", 0.40, 0.90, float(st.session_state.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)), 0.01)
    st.caption("Model confidence measures the classifier's probability estimate, not real-world certainty.")
    st.divider()
    st.markdown("#### Destructive action")
    confirm = st.checkbox("I understand this permanently clears local SQLite emails and feedback history.")
    if st.button("Delete all stored records", disabled=not confirm):
        delete_all()
        st.success("Stored emails and feedback history were deleted.")
        st.rerun()
