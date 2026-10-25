from __future__ import annotations

import streamlit as st

from modules.database import list_emails, mark_reviewed
from views.common import page_header, render_record_details


def render() -> None:
    page_header("Human in the loop", "Manual review", "Resolve low-confidence or conflicting suggestions without overwriting the model record.")
    records = [r for r in list_emails() if r.get("review_status") == "Pending review" or r.get("needs_review")]
    if not records:
        st.success("The manual-review queue is clear.")
        st.caption("New messages enter the queue when confidence is low, the input is unusually short, a deadline is uncertain, or a high-signal rule changes priority.")
        return
    st.warning(f"{len(records)} email(s) need a human decision.")
    labels = [f"{r.get('email_id')} · {r.get('subject','')[:80]}" for r in records]
    index = st.selectbox("Select a review item", range(len(labels)), format_func=lambda i: labels[i])
    record = records[index]
    st.markdown("#### Why this item is here")
    reasons = record.get("review_reasons", []) or []
    if not reasons:
        reasons = ["The record was explicitly requested for review or has a pending review state."]
    for reason in reasons:
        st.markdown(f"- {reason}")
    render_record_details(record, key_prefix="review_detail")
    if st.button("Mark reviewed without changing labels", type="primary", key="review_complete"):
        mark_reviewed(record["email_id"], True)
        st.success("Review decision recorded.")
        st.rerun()
