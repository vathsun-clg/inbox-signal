from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.database import feedback_records
from views.common import page_header


def render() -> None:
    page_header("Accountability", "Feedback history", "A durable audit trail of model suggestions and human corrections.")
    records = feedback_records()
    if not records:
        st.info("No human corrections have been recorded yet. Review an email and save a correction to create history.")
        return
    df = pd.DataFrame(records)
    st.metric("Recorded decisions", len(df))
    st.dataframe(df[[col for col in ["created_at", "email_id", "original_priority", "final_priority", "user_priority", "original_intent", "user_intent", "review_status", "correction_notes"] if col in df.columns]], use_container_width=True, hide_index=True)
    st.download_button("Download feedback history CSV", df.to_csv(index=False).encode("utf-8"), "feedback_history.csv", "text/csv")
    st.caption("Corrections are preserved as feedback history. They do not become training data unless a future, explicit retraining workflow prepares and approves them.")
