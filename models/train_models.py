from __future__ import annotations

import json
from pathlib import Path

import joblib

from config import MODEL_DIR, SYNTHETIC_DATA_PATH
from modules.evaluation import build_pipeline, load_training_data, training_summary


def train_and_save() -> dict:
    df = load_training_data(SYNTHETIC_DATA_PATH)
    if len(df) < 14:
        raise ValueError("The training dataset is too small to train reliable demo models.")
    for target in ("priority", "intent"):
        if df[target].nunique() < 2:
            raise ValueError(f"The {target} dataset must contain at least two classes.")
        for model_name in ("multinomial_nb", "logistic_regression"):
            pipeline = build_pipeline(model_name)
            pipeline.fit(df["text"], df[target])
            joblib.dump(pipeline, MODEL_DIR / f"{target}_{model_name}.joblib")
    summary = training_summary(df)
    (MODEL_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = train_and_save()
    print(json.dumps(result, indent=2))
