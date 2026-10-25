from __future__ import annotations

import streamlit as st

from config import DEFAULT_ACTIVE_MODEL, DEFAULT_CONFIDENCE_THRESHOLD, MODEL_DIR
from modules.database import init_db
from models.train_models import train_and_save
from views import analyze, dashboard, evaluation, feedback, inbox, review, settings
from views.common import inject_css

st.set_page_config(page_title="Inbox Signal", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
inject_css()
init_db()

if not (MODEL_DIR / "priority_logistic_regression.joblib").exists():
    with st.spinner("Training the local demo models for the first run…"):
        train_and_save()

if "active_model" not in st.session_state:
    st.session_state["active_model"] = DEFAULT_ACTIVE_MODEL
if "confidence_threshold" not in st.session_state:
    st.session_state["confidence_threshold"] = DEFAULT_CONFIDENCE_THRESHOLD

with st.sidebar:
    st.markdown("<div class='brand-mark'><div class='brand-glyph'></div><div><div class='brand-title'>inbox / signal</div><div class='brand-sub'>local triage desk</div></div></div>", unsafe_allow_html=True)
    st.caption("Suggestions, not verdicts.")
    page = st.radio(
        "Workspace",
        ["Dashboard", "Analyze email", "Email inbox", "Manual review", "Evaluation", "Feedback history", "Settings & data"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("<div class='mini-label'>Active model</div>", unsafe_allow_html=True)
    st.caption("Logistic Regression" if st.session_state["active_model"] == "logistic_regression" else "Multinomial Naive Bayes")
    st.caption(f"Review threshold · {st.session_state['confidence_threshold']:.0%}")
    st.divider()
    st.caption("Local-only · synthetic/public samples · human confirmation required")

try:
    if page == "Dashboard":
        dashboard.render()
    elif page == "Analyze email":
        analyze.render()
    elif page == "Email inbox":
        inbox.render()
    elif page == "Manual review":
        review.render()
    elif page == "Evaluation":
        evaluation.render()
    elif page == "Feedback history":
        feedback.render()
    elif page == "Settings & data":
        settings.render()
except Exception as exc:
    st.error("The application hit an unexpected error while rendering this page.")
    st.exception(exc)
