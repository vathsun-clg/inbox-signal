from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.evaluation import compare_models, load_training_data, training_summary
from views.common import page_header


def render() -> None:
    page_header("Model transparency", "Evaluation & performance", "Held-out test results from the synthetic dataset. These are indicators, not production guarantees.")
    try:
        df = load_training_data()
        summary = training_summary(df)
    except Exception as exc:
        st.error(f"Could not load the training dataset: {exc}")
        return
    a, b, c = st.columns(3)
    a.metric("Training rows", summary["rows"])
    b.metric("Priority classes", len(summary["priority_distribution"]))
    c.metric("Intent classes", len(summary["intent_distribution"]))
    st.info("The model is trained on synthetic examples. Test performance can differ materially from real organizational email, especially under class imbalance and domain shift.")
    target = st.selectbox("Evaluate label", ["priority", "intent"], format_func=lambda x: "Priority" if x == "priority" else "Intent")
    try:
        table, details = compare_models(target)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        return
    st.dataframe(table.style.format({"Accuracy": "{:.1%}", "Macro F1": "{:.1%}", "Weighted F1": "{:.1%}", "Macro Precision": "{:.1%}", "Macro Recall": "{:.1%}"}), use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspect confusion matrix", list(details.keys()), format_func=lambda x: x.replace("_", " ").title())
    detail = details[selected]
    matrix = pd.DataFrame(detail["confusion_matrix"], index=detail["labels"], columns=detail["labels"])
    st.markdown(f"#### {selected.replace('_', ' ').title()} · {target.title()}")
    left, right = st.columns([1, 1])
    with left:
        fig = px.imshow(matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Rows"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        report = pd.DataFrame(detail["report"]).T.reset_index().rename(columns={"index": "Class"})
        cols = [col for col in ["Class", "precision", "recall", "f1-score", "support"] if col in report.columns]
        st.dataframe(report[cols].style.format({col: "{:.1%}" for col in ["precision", "recall", "f1-score"] if col in report.columns}), use_container_width=True, hide_index=True)
        st.caption(f"Train rows: {detail['train_size']} · Held-out test rows: {detail['test_size']}")
    with st.expander("Dataset and generalization notes"):
        st.write("Class distribution — priority:", summary["priority_distribution"])
        st.write("Class distribution — intent:", summary["intent_distribution"])
        st.write("The synthetic corpus is useful for reproducible local demos and regression tests, not for claiming production readiness. Review real, authorized samples before retraining. Human feedback is stored separately and is never silently used as new training data.")
