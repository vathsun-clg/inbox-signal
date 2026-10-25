from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib

from config import DEFAULT_ACTIVE_MODEL, MODEL_DIR, PRIORITIES
from modules.urgency_detection import detect_urgency


def _artifact_path(task: str, model_name: str) -> Path:
    return MODEL_DIR / f"{task}_{model_name}.joblib"


def load_model(task: str, model_name: str = DEFAULT_ACTIVE_MODEL):
    path = _artifact_path(task, model_name)
    if not path.exists():
        from models.train_models import train_and_save
        train_and_save()
    if not path.exists():
        raise FileNotFoundError(f"Model artifact is missing: {path.name}")
    return joblib.load(path)


def predict_with_confidence(text: str, task: str, model_name: str = DEFAULT_ACTIVE_MODEL) -> Dict[str, Any]:
    model = load_model(task, model_name)
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    best_index = int(probabilities.argmax())
    return {
        "label": str(classes[best_index]),
        "confidence": float(probabilities[best_index]),
        "probabilities": {str(label): float(prob) for label, prob in zip(classes, probabilities)},
    }


def adjust_priority(model_prediction: Dict[str, Any], text: str, urgency: Dict[str, Any] | None = None) -> Dict[str, Any]:
    urgency = urgency or detect_urgency(text)
    original = model_prediction["label"]
    final = original
    reasons = []
    evidence = list(urgency.get("evidence", []))
    if urgency.get("urgency_level") == "High" and original in {"Low", "Medium"}:
        final = "High" if original == "Low" else "Critical"
        reasons.append("High-signal urgency evidence raised the model suggestion.")
    elif urgency.get("deadline_phrases") and original == "Low":
        final = "Medium"
        reasons.append("A possible deadline raised a low-priority model suggestion to medium for review.")
    if any("security" in reason for reason in urgency.get("reasons", [])) and original != "Critical":
        final = "Critical"
        reasons.append("A security-incident signal requires explicit human review at critical priority.")
    if final not in PRIORITIES:
        final = original
    return {
        "original_priority": original,
        "final_priority": final,
        "rule_changed": final != original,
        "rule_reason": " ".join(reasons),
        "evidence": evidence,
        "confidence": model_prediction.get("confidence"),
    }


def classify_email(processed_text: str, model_name: str = DEFAULT_ACTIVE_MODEL, confidence_threshold: float = 0.62) -> Dict[str, Any]:
    priority_prediction = predict_with_confidence(processed_text, "priority", model_name)
    intent_prediction = predict_with_confidence(processed_text, "intent", model_name)
    urgency = detect_urgency(processed_text)
    priority = adjust_priority(priority_prediction, processed_text, urgency)
    needs_review = (
        priority_prediction["confidence"] < confidence_threshold
        or intent_prediction["confidence"] < confidence_threshold
        or len(processed_text.split()) < 6
        or bool(urgency.get("deadline_uncertain"))
        or (priority["rule_changed"] and urgency.get("urgency_level") == "High")
    )
    review_reasons = []
    if priority_prediction["confidence"] < confidence_threshold or intent_prediction["confidence"] < confidence_threshold:
        review_reasons.append("One or more model confidences are below the configured threshold.")
    if len(processed_text.split()) < 6:
        review_reasons.append("The message is unusually short.")
    if urgency.get("deadline_uncertain"):
        review_reasons.append("A deadline phrase may need user confirmation.")
    if priority["rule_changed"] and urgency.get("urgency_level") == "High":
        review_reasons.append("A business rule changed the priority based on high urgency evidence.")
    return {
        "model_name": model_name,
        "original_priority": priority["original_priority"],
        "final_priority": priority["final_priority"],
        "original_intent": intent_prediction["label"],
        "final_intent": intent_prediction["label"],
        "confidence": priority_prediction["confidence"],
        "intent_confidence": intent_prediction["confidence"],
        "priority_probabilities": priority_prediction["probabilities"],
        "intent_probabilities": intent_prediction["probabilities"],
        "urgency": urgency,
        "rule_changed": priority["rule_changed"],
        "rule_reason": priority["rule_reason"],
        "rule_evidence": priority["evidence"],
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "classified_at": datetime.now().isoformat(timespec="seconds"),
    }
