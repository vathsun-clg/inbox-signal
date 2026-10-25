from modules.database import upsert_email
from modules.email_service import analyze_records, load_synthetic_samples

df, errors = load_synthetic_samples()
records, analysis_errors = analyze_records(df.to_dict("records"), mask_enabled=True, model_name="logistic_regression", confidence_threshold=0.62)
for record in records:
    upsert_email(record)
print({"imported": len(records), "validation_errors": errors, "analysis_errors": analysis_errors})
