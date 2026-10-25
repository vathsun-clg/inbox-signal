from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from config import INTENTS, PRIORITIES
from modules.database import list_emails, mark_reviewed, upsert_email
from modules.email_service import analyze_record
from views.common import dataframe_for_display, page_header, render_record_details


def render() -> None:
    page_header("Queue", "Email inbox", "Search and sort the stored queue. Open a record when you need the evidence behind a suggestion.")
    records = list_emails()
    if not records:
        st.info("No emails are stored yet. Analyze a message or load synthetic samples from Settings.")
        return
    with st.expander("Filters and sorting", expanded=True):
        row1 = st.columns([2, 1, 1, 1])
        query = row1[0].text_input("Search subject and content", placeholder="Search…")
        priority = row1[1].selectbox("Priority", ["All"] + PRIORITIES)
        intent = row1[2].selectbox("Intent", ["All"] + INTENTS)
        review = row1[3].selectbox("Review", ["All", "Pending review", "Reviewed", "Not required"])
        row2 = st.columns([1, 1, 1, 1])
        confidence_min = row2[0].slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        sort_by = row2[1].selectbox("Sort by", ["Received date", "Priority", "Confidence"])
        sort_order = row2[2].selectbox("Order", ["Newest / highest first", "Oldest / lowest first"])
        date_from = row2[3].date_input("Received from", value=None)
    filtered = []
    for record in records:
        haystack = f"{record.get('subject','')} {record.get('body','')}".lower()
        received = pd.to_datetime(record.get("received_at"), errors="coerce")
        if query and query.lower() not in haystack:
            continue
        if priority != "All" and record.get("final_priority") != priority:
            continue
        if intent != "All" and record.get("final_intent") != intent:
            continue
        if review != "All" and record.get("review_status") != review:
            continue
        if float(record.get("confidence") or 0) < confidence_min:
            continue
        if date_from and not pd.isna(received) and received.date() < date_from:
            continue
        filtered.append(record)
    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    if sort_by == "Priority":
        filtered.sort(key=lambda r: priority_order.get(r.get("final_priority"), 9), reverse=sort_order == "Oldest / lowest first")
    elif sort_by == "Confidence":
        filtered.sort(key=lambda r: float(r.get("confidence") or 0), reverse=sort_order == "Newest / highest first")
    else:
        filtered.sort(key=lambda r: str(r.get("received_at", "")), reverse=sort_order == "Newest / highest first")
    st.caption(f"Showing {len(filtered)} of {len(records)} stored records.")
    st.dataframe(dataframe_for_display(filtered), use_container_width=True, hide_index=True)
    selected = None
    if filtered:
        labels = [f"{r.get('email_id')} · {r.get('subject','')[:70]}" for r in filtered]
        selected_index = st.selectbox("Open a record", range(len(labels)), format_func=lambda idx: labels[idx])
        selected = filtered[selected_index]
    if selected:
        st.divider()
        render_record_details(selected, key_prefix="inbox_detail")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Mark as reviewed", key="inbox_mark_reviewed"):
                mark_reviewed(selected["email_id"], True)
                st.success("Marked as reviewed.")
                st.rerun()
        with b2:
            if st.button("Reanalyze with current text", key="inbox_reanalyze"):
                try:
                    fresh = analyze_record(selected, mask_enabled=selected.get("masked", True), model_name=selected.get("model_name", "logistic_regression"))
                    upsert_email(fresh)
                    st.success("Record reanalyzed and saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Reanalysis failed: {exc}")
