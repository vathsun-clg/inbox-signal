# Inbox Signal — Intelligent Email Priority Classification

Inbox Signal is a local-first Streamlit application for transparent email triage. It analyzes approved synthetic, public, or explicitly authorized email samples and produces suggestions for intent, priority, urgency, possible deadlines, action items, confidence, and manual review. It keeps model predictions, rule-based adjustments, and human corrections as separate fields.

This is decision support, not an autonomous mailbox agent. The app does not connect to a real mailbox, send email, execute instructions found in email content, call external LLMs, or create external tasks automatically.

## Features

The application includes a live dashboard backed by SQLite; pasted-email analysis with inspectable preprocessing; optional masking of email addresses and phone numbers; conservative quoted-reply and signature handling; urgency and deadline evidence; action-item candidates that can be accepted, rejected, or edited; a searchable/filterable inbox; a manual-review queue; held-out evaluation comparing Multinomial Naive Bayes and Logistic Regression; persistent feedback history; focused CSV exports; and safe import/delete controls.

The core pipeline is Python, Pandas, regular expressions, standard date parsing, scikit-learn TF-IDF, MultinomialNB, LogisticRegression, Plotly, SQLite, and CSV. It requires no paid API, cloud AI service, secret, or external mailbox credential.

## Architecture

`app.py` owns Streamlit configuration, navigation, and shared state. `views/` contains the seven page views. `modules/preprocessing.py`, `privacy.py`, `urgency_detection.py`, and `action_extraction.py` keep text processing and business signals testable. `modules/email_service.py` validates CSV input and orchestrates analysis. `modules/database.py` owns SQLite schema and persistence. `modules/evaluation.py` owns held-out evaluation. `models/train_models.py` trains and saves artifacts, while `models/email_classifier.py` reloads artifacts and produces predictions plus separate rule-based adjustments. The permitted synthetic corpus lives in `data/synthetic_emails.csv`.

## Python version and installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the local models

Training is reproducible and uses the bundled synthetic dataset. The command fits separate priority and intent pipelines for both supported model types and saves them under `models/saved_models/`.

```bash
python -m models.train_models
```

The first Streamlit run will also train missing artifacts automatically with a visible first-run spinner.

## Initialize the database

Database initialization is automatic when `app.py` starts. To create the empty SQLite schema without opening Streamlit:

```bash
python -c "from modules.database import init_db; init_db(); print('SQLite ready')"
```

The local database is `data/email_triage.sqlite3` and is intentionally ignored by Git.

## Start the application

For local development:

```bash
streamlit run app.py
```

For a managed preview or a headless environment:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 3000 --server.headless true --server.fileWatcherType none
```

Open the URL printed by Streamlit. Use Settings & data to load the bundled synthetic dataset, or start with Analyze email.

## Run tests

```bash
pytest -q
```

The test suite covers preprocessing, privacy masking, urgency negation and deadlines, action extraction, model predictions, low-confidence review flags, SQLite persistence and corrections, CSV validation, exports, and invalid inputs.

The executed suite output and actual held-out metrics are recorded in `TEST_RESULTS.md`.

## CSV format

Required columns are:

```text
email_id,subject,body,received_at
```

`received_at` must parse as a timestamp. `email_id` must be non-empty and unique within an upload. The Settings page validates each row, reports invalid rows, and continues with valid rows. Existing IDs are updated safely rather than silently duplicated.

A sample CSV is available as `data/synthetic_emails.csv` and through the Download sample CSV button. Training-only columns `priority` and `intent` are included in the bundled corpus; uploaded operational samples only need the four required fields.

## Model and business-rule behavior

The active model predicts priority and intent from TF-IDF features. Confidence is the highest class probability and must not be read as real-world certainty. The app separately looks for high-signal contexts such as a production outage, security incident, immediate approval, or an explicit time constraint. A rule may raise a low or medium suggestion, but the original model output remains visible. Messages that say “not urgent” are not treated as urgent solely because they contain the word urgent.

Manual review is requested for low confidence, unusually short messages, uncertain deadlines, and high-signal rule changes. Users can override labels, confirm/reject action candidates, and add notes. Corrections are stored in feedback history and never silently become training data.

## Privacy and security

Use only synthetic, public, or explicitly authorized samples. Email text is untrusted input and is never executed. Privacy masking is available before analysis/storage for common email-address and phone-number patterns. The masking is intentionally conservative: it may miss obfuscated or international formats and may occasionally match a non-phone numeric string. Quoted-reply and signature removal also use heuristics and must be inspected in the Analyze page before consequential use. Summary exports omit full bodies by default.

No credentials are required. Keep any future configuration or secrets outside source code. Review logs and exported files before sharing them because user-entered text can still contain sensitive information.

## Deployment options

The application is designed for local use and can be run on any Python-capable host that can install the pinned dependencies. Streamlit Community Cloud can run it from a Git repository: commit the project, set the main file to `app.py`, and let the host install `requirements.txt`. For a private internal deployment, run the headless command behind the organization's normal HTTPS reverse proxy and access controls. Do not expose a mailbox integration or public email content without a separate authorization and security review.

The managed preview in this project uses Streamlit on port 3000. A deployment platform that expects a container should run the same command and expose port 3000.

## Troubleshooting

If model artifacts are missing, run `python -m models.train_models`. If imports fail, activate the virtual environment and run `pip install -r requirements.txt`. If a CSV import reports invalid rows, inspect the row-level table for missing columns, empty subject/body, duplicate IDs, or invalid timestamps. If a page shows no data, load the synthetic dataset from Settings & data. If the database becomes corrupted during local experimentation, stop the app, remove `data/email_triage.sqlite3`, and restart; this clears local records only.

## Known limitations

The training data is synthetic and intentionally small. Held-out scores are useful for regression checks, not a production readiness claim. Real-world language, organizational jargon, multilingual mail, adversarial text, attachments, phishing, and context outside the message body can change results. Deadline interpretation is deliberately conservative and may ask for confirmation. The system does not verify whether an email is authentic, safe, or truly consequential. Human review remains required for consequential decisions.
