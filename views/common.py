from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import pandas as pd
import streamlit as st

from config import INTENTS, PRIORITIES, PRIORITY_COLORS
from modules.database import get_email, mark_reviewed, update_actions, update_correction


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#19202b; --muted:#6b7280; --line:#e5e7eb; --surface:#fbfaf8; --indigo:#4f46e5; }
        .stApp { background: #f7f6f3; color: var(--ink); }
        [data-testid="stSidebar"] { background: #141922; border-right: 1px solid #252c38; }
        [data-testid="stSidebar"] * { color: #e7eaf0 !important; }
        .brand-mark { display:flex; gap:10px; align-items:center; padding: 8px 0 22px; }
        .brand-glyph { width:32px; height:32px; border:2px solid #8f91ff; border-radius:9px; position:relative; }
        .brand-glyph:before { content:''; position:absolute; width:9px; height:9px; background:#8f91ff; border-radius:50%; top:9px; left:9px; box-shadow: 0 0 0 4px rgba(143,145,255,.14); }
        .brand-title { font-size: 15px; line-height:1.05; font-weight:800; letter-spacing:-.03em; }
        .brand-sub { font-size:10px; color:#9ca3af !important; letter-spacing:.13em; text-transform:uppercase; margin-top:3px; }
        .eyebrow { font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:#6b7280; font-weight:700; margin-bottom:6px; }
        .page-title { font-size:36px; line-height:1.02; letter-spacing:-.05em; font-weight:800; margin: 0 0 8px; color:#19202b; }
        .page-subtitle { color:#6b7280; font-size:15px; margin-bottom:24px; }
        .metric-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:17px 18px; min-height:112px; box-shadow:0 4px 18px rgba(25,32,43,.035); }
        .metric-label { color:#6b7280; font-size:12px; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
        .metric-value { font-size:29px; line-height:1; font-weight:800; margin:12px 0 6px; letter-spacing:-.05em; }
        .metric-note { color:#8a919c; font-size:12px; }
        .signal-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px; box-shadow:0 4px 18px rgba(25,32,43,.035); }
        .signal-card h4 { margin:0 0 3px; font-size:16px; }
        .signal-card p { color:#6b7280; font-size:13px; margin:0 0 14px; }
        .priority-pill { display:inline-flex; align-items:center; gap:6px; padding:4px 9px; border-radius:999px; font-size:11px; font-weight:800; }
        .status-pill { display:inline-flex; align-items:center; padding:4px 9px; border-radius:999px; font-size:11px; font-weight:700; background:#eef0f4; color:#4b5563; }
        .evidence { background:#f7f7fb; border-left:3px solid #8f91ff; padding:10px 12px; border-radius:0 10px 10px 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; color:#4b5563; }
        .mini-label { font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.1em; font-weight:700; }
        div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:12px 14px; border-radius:14px; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f"<div class='eyebrow'>{eyebrow}</div><div class='page-title'>{title}</div><div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def priority_badge(value: str) -> str:
    color = PRIORITY_COLORS.get(value, "#6b7280")
    return f"<span class='priority-pill' style='background:{color}18;color:{color}'>{value}</span>"


def confidence_bar(value: float | None) -> None:
    value = float(value or 0)
    st.progress(max(0.0, min(1.0, value)), text=f"{value:.0%} model confidence")


def dataframe_for_display(records: list[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "Email ID": record.get("email_id"),
                "Subject": record.get("subject"),
                "Received": str(record.get("received_at", ""))[:16].replace("T", " "),
                "Intent": record.get("final_intent"),
                "Priority": record.get("final_priority"),
                "Confidence": f"{float(record.get('confidence') or 0):.0%}",
                "Urgency": record.get("urgency_level"),
                "Actions": len(record.get("actions", [])),
                "Review": record.get("review_status"),
            }
        )
    return pd.DataFrame(rows)


def render_record_details(record: Dict[str, Any], key_prefix: str = "detail") -> None:
    st.markdown(f"### {record.get('subject') or '(No subject)'}")
    st.caption(f"{record.get('email_id')} · received {record.get('received_at')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(priority_badge(record.get("final_priority")), unsafe_allow_html=True)
    c2.metric("Intent", record.get("final_intent") or "—")
    c3.metric("Confidence", f"{float(record.get('confidence') or 0):.0%}")
    c4.metric("Urgency", record.get("urgency_level") or "Low")
    tabs = st.tabs(["Original", "Processed text", "Signals & actions", "Human correction"])
    with tabs[0]:
        st.text_area("Original email", value=record.get("body", ""), height=200, disabled=True, key=f"{key_prefix}_original")
    with tabs[1]:
        st.text_area("Text sent to the local classifier", value=record.get("processed_text", ""), height=180, disabled=True, key=f"{key_prefix}_processed")
        if record.get("masked"):
            st.info("Privacy masking was enabled before classification/storage.")
    with tabs[2]:
        if record.get("rule_changed"):
            st.warning(f"Rule-based adjustment: {record.get('rule_reason')}")
        elif record.get("rule_reason"):
            st.info(record.get("rule_reason"))
        st.markdown("**Evidence**")
        evidence = record.get("urgency_evidence", []) or []
        if evidence:
            for item in evidence:
                st.markdown(f"<div class='evidence'>{item}</div>", unsafe_allow_html=True)
        else:
            st.caption("No explicit urgency evidence detected.")
        st.markdown("**Possible deadlines**")
        st.write(record.get("deadline_phrases") or "No deadline phrase detected.")
        st.markdown("**Potential action items**")
        actions = record.get("actions", []) or []
        if not actions:
            st.caption("No action candidates extracted.")
        updated_actions = []
        for index, action in enumerate(actions):
            left, right = st.columns([4, 1])
            with left:
                description = st.text_input("Action", value=action.get("description", ""), key=f"{key_prefix}_action_{index}", label_visibility="collapsed")
                st.caption(f"Evidence: {action.get('source_text', '')} · {action.get('confidence', 'Unknown')} confidence")
            with right:
                status = st.selectbox("Status", ["Pending", "Accepted", "Rejected"], index=["Pending", "Accepted", "Rejected"].index(action.get("status", "Pending")) if action.get("status", "Pending") in ["Pending", "Accepted", "Rejected"] else 0, key=f"{key_prefix}_status_{index}")
            updated = dict(action)
            updated["description"] = description
            updated["status"] = status
            updated_actions.append(updated)
        if actions and st.button("Save action decisions", key=f"{key_prefix}_save_actions"):
            update_actions(record["email_id"], updated_actions)
            st.success("Action decisions saved.")
    with tabs[3]:
        with st.form(f"{key_prefix}_correction_form"):
            chosen_priority = st.selectbox("Human priority override", ["(keep suggestion)"] + PRIORITIES, index=(["(keep suggestion)"] + PRIORITIES).index(record.get("user_priority") or "(keep suggestion)"))
            chosen_intent = st.selectbox("Human intent override", ["(keep suggestion)"] + INTENTS, index=(["(keep suggestion)"] + INTENTS).index(record.get("user_intent") or "(keep suggestion)"))
            notes = st.text_area("Correction notes", value=record.get("correction_notes") or "", placeholder="Why should this suggestion change?")
            submitted = st.form_submit_button("Save correction", type="primary")
        if submitted:
            update_correction(record["email_id"], None if chosen_priority == "(keep suggestion)" else chosen_priority, None if chosen_intent == "(keep suggestion)" else chosen_intent, notes)
            st.success("Human correction saved to feedback history.")
            st.rerun()


def select_record(records: list[Dict[str, Any]], label: str = "Open an email") -> Dict[str, Any] | None:
    if not records:
        st.info("No emails are available yet. Analyze an email or load the synthetic dataset to begin.")
        return None
    labels = [f"{r.get('email_id')} · {r.get('subject', '')[:70]}" for r in records]
    choice = st.selectbox(label, range(len(labels)), format_func=lambda index: labels[index])
    return records[choice]
