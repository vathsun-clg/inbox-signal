from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import SYNTHETIC_DATA_PATH


def load_training_data(path=SYNTHETIC_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ("subject", "body", "priority", "intent"):
        if column not in df.columns:
            raise ValueError(f"Training dataset is missing the required column: {column}")
    df["text"] = (df["subject"].fillna("") + "\n" + df["body"].fillna("")).str.strip()
    return df


def build_pipeline(model_name: str) -> Pipeline:
    estimator = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear") if model_name == "logistic_regression" else MultinomialNB()
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)), ("model", estimator)])


def _split(df: pd.DataFrame, target: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    try:
        return train_test_split(df["text"], df[target], test_size=0.25, random_state=42, stratify=df[target])
    except ValueError:
        return train_test_split(df["text"], df[target], test_size=0.25, random_state=42)


def evaluate_model(df: pd.DataFrame, target: str, model_name: str) -> Dict[str, Any]:
    x_train, x_test, y_train, y_test = _split(df, target)
    pipeline = build_pipeline(model_name)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    labels = sorted(df[target].dropna().unique().tolist())
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    return {
        "target": target,
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "labels": labels,
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
    }


def compare_models(target: str = "priority") -> tuple[pd.DataFrame, Dict[str, Any]]:
    df = load_training_data()
    results = [evaluate_model(df, target, model_name) for model_name in ("multinomial_nb", "logistic_regression")]
    table = pd.DataFrame(
        [
            {
                "Model": result["model_name"].replace("_", " ").title(),
                "Accuracy": result["accuracy"],
                "Macro F1": result["macro_f1"],
                "Weighted F1": result["weighted_f1"],
                "Macro Precision": result["macro_precision"],
                "Macro Recall": result["macro_recall"],
                "Test rows": result["test_size"],
            }
            for result in results
        ]
    )
    return table, {result["model_name"]: result for result in results}


def training_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": len(df),
        "priority_distribution": df["priority"].value_counts().to_dict(),
        "intent_distribution": df["intent"].value_counts().to_dict(),
        "unique_subjects": df["subject"].nunique(),
    }
