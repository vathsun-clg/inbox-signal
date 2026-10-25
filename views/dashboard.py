from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import INTENT_COLORS, PRIORITY_COLORS
from modules.database import counts, list_emails
from views.common import page_header


def render() -> None:
    page_header("Overview", "Triage dashboard", "A live view of the signal in your local email queue.")
    records = list_emails()
    summary = counts()
    if not records:
        st.info("Your queue is empty. Use Analyze Email to process a message or Settings to load the built-in synthetic dataset.")
        return
    metric_specs = [
        ("Total emails", summary["total"], "stored records"),
        ("Critical", summary["critical"], "rule-adjusted or model-suggested"),
        ("High priority", summary["high"], "needs attention"),
        ("Awaiting review", summary["review"], "human decisions pending"),
        ("Deadlines", summary["deadlines"], "possible time constraints"),
        ("Pending actions", summary["actions"], "candidate actions"),
    ]
    cols = st.columns(len(metric_specs))
    for col, (label, value, note) in zip(cols, metric_specs):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)
    st.write("")
    df = pd.DataFrame(records)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Priority distribution")
        priority_df = df["final_priority"].value_counts().rename_axis("Priority").reset_index(name="Emails")
        fig = px.bar(priority_df, x="Priority", y="Emails", color="Priority", color_discrete_map=PRIORITY_COLORS, template="simple_white")
        fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Intent distribution")
        intent_df = df["final_intent"].value_counts().rename_axis("Intent").reset_index(name="Emails")
        fig = px.bar(intent_df, x="Emails", y="Intent", orientation="h", color="Intent", color_discrete_map=INTENT_COLORS, template="simple_white")
        fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    lower_left, lower_right = st.columns([1.2, 0.8])
    with lower_left:
        st.markdown("#### Received over time")
        timeline = df.assign(received=pd.to_datetime(df["received_at"], errors="coerce")).dropna(subset=["received"])
        timeline = timeline.groupby(timeline["received"].dt.date).size().reset_index(name="Emails")
        fig = px.area(timeline, x="received", y="Emails", markers=True, template="simple_white")
        fig.update_traces(line_color="#4F46E5", fillcolor="rgba(79,70,229,.13)")
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with lower_right:
        st.markdown("#### Confidence and review")
        conf_df = df[["confidence", "review_status"]].copy()
        conf_df["Confidence"] = conf_df["confidence"].astype(float)
        fig = px.histogram(conf_df, x="Confidence", color="review_status", nbins=8, barmode="stack", template="simple_white", color_discrete_sequence=["#4F46E5", "#F08C46", "#9CA3AF"])
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("All dashboard values are calculated from the current SQLite records. Model suggestions are not guaranteed facts.")
