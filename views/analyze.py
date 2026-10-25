from __future__ import annotations

from datetime import datetime, time

import streamlit as st

from config import DEFAULT_ACTIVE_MODEL, DEFAULT_CONFIDENCE_THRESHOLD
from modules.database import upsert_email
from modules.email_service import analyze_record
from modules.preprocessing import preprocess_email
from views.common import confidence_bar, page_header, priority_badge


def render() -> None:
    page_header("New analysis", "Analyze an email", "Paste a message, inspect what the local pipeline sees, then save the suggestion to the queue.")
    left, right = st.columns([1.15, 0.85])
    with left:
        with st.form("analyze_email_form"):
            subject = st.text_input("Subject", placeholder="e.g. Production API is down")
            body = st.text_area("Body", height=230, placeholder="Paste the approved email text here…")
            received_date = st.date_input("Received date", value=datetime.now().date())
            received_time = st.time_input("Received time", value=time(hour=datetime.now().hour, minute=datetime.now().minute))
            email_id = st.text_input("Email ID", placeholder="Optional; a local ID will be generated if blank")
            st.markdown("**Privacy and model controls**")
            mask_enabled = st.checkbox("Mask common email addresses and phone numbers before analysis/storage", value=True)
            model_name = st.selectbox("Active model", ["logistic_regression", "multinomial_nb"], format_func=lambda x: "Logistic Regression" if x == "logistic_regression" else "Multinomial Naive Bayes")
            threshold = st.slider("Manual review confidence threshold", 0.40, 0.90, float(st.session_state.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)), 0.01)
            submitted = st.form_submit_button("Analyze and save suggestion", type="primary")
    with right:
        st.markdown("#### Before classification")
        preview = preprocess_email(subject, body)
        st.text_area("Inspectable processed text", preview["processed_text"], height=180, disabled=True)
        if preview["warnings"]:
            for warning in preview["warnings"]:
                st.warning(warning)
        else:
            st.success("The message is long enough for a normal first-pass analysis.")
        st.caption("Signature and quoted-reply detection is conservative. Review the processed text before relying on a suggestion.")
    if submitted:
        if not subject.strip() or not body.strip():
            st.error("Add both a subject and a body before analyzing.")
            return
        try:
            with st.spinner("Running local preprocessing and model inference…"):
                received_at = datetime.combine(received_date, received_time)
                record = analyze_record({"email_id": email_id, "subject": subject, "body": body, "received_at": received_at.isoformat()}, mask_enabled=mask_enabled, model_name=model_name, confidence_threshold=threshold)
                upsert_email(record)
            st.session_state["confidence_threshold"] = threshold
            st.success(f"Saved {record['email_id']} to the email queue.")
            st.markdown(f"### Suggested result {priority_badge(record['final_priority'])}", unsafe_allow_html=True)
            st.write(f"Intent: **{record['final_intent']}** · urgency: **{record['urgency_level']}**")
            confidence_bar(record["confidence"])
            if record.get("rule_changed"):
                st.warning(f"Rule-based adjustment: {record.get('rule_reason')}")
            if record.get("needs_review"):
                st.info("This email is in the manual-review queue because the model or business-rule signals are uncertain.")
            if record.get("deadline_phrases"):
                st.write("Possible deadline phrase(s):", record["deadline_phrases"])
        except Exception as exc:
            st.error(f"Could not analyze this message: {exc}")
