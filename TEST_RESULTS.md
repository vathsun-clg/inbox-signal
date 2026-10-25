# Verification Results

## Automated checks

Executed from the project root on 2026-10-17:

```text
PYTHONPATH=. .venv/bin/pytest -q
........                                                                 [100%]
8 passed, 24 warnings in 1.77s
```

The warnings are NumPy/joblib deprecation warnings during loading of persisted model artifacts; they do not fail the suite. A `compileall` pass also completed successfully, and `pip check` reported `No broken requirements found`.

## Training run

```text
PYTHONPATH=. .venv/bin/python -m models.train_models
rows: 40
priority_distribution: Medium 12, High 11, Low 9, Critical 8
intent_distribution: Complaint 8, Request 8, Notification 7, Approval 6, Information 6, Meeting 5
```

## Held-out evaluation

The evaluation uses a reproducible random seed and a 25% held-out split per target. These metrics are actual predictions from the current synthetic dataset and are not production claims.

| Target | Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Test rows |
|---|---|---:|---:|---:|---:|---:|---:|
| Priority | Multinomial Naive Bayes | 40.0% | 35.5% | 34.6% | 31.3% | 41.7% | 10 |
| Priority | Logistic Regression | 50.0% | 42.5% | 43.0% | 39.2% | 50.0% | 10 |
| Intent | Multinomial Naive Bayes | 30.0% | 25.0% | 20.0% | 22.2% | 33.3% | 10 |
| Intent | Logistic Regression | 50.0% | 46.7% | 42.0% | 43.1% | 58.3% | 10 |

The app exposes the full class-wise precision/recall/F1 report and confusion matrix in the Evaluation page. Because the corpus is small and synthetic, these results should be used for regression checks and transparency, not for production readiness claims.

## Preview smoke test

The Streamlit service listened successfully on port 3000 and returned HTTP 200 for the preview origin. A screenshot check confirmed the dashboard renders both the empty state and the populated dashboard backed by 40 analyzed synthetic records.
