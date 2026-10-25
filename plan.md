# Intelligent Email Priority Classification — Implementation Plan

## Scope

Build a local-first Streamlit application that classifies approved sample email text into intent and priority suggestions, detects urgency and possible deadlines, extracts potential action items, persists records and human corrections in SQLite, and exposes transparent review and evaluation workflows. The core system uses only local Python libraries and synthetic data; it never connects to a mailbox or sends external actions.

## Architecture

- `app.py`: Streamlit shell, navigation, shared session state, page rendering, and user-facing error boundaries.
- `config.py`: paths, label sets, thresholds, supported CSV fields, and application configuration.
- `modules/preprocessing.py`: text normalization, quoted-reply/signature handling, short-text warnings, and inspectable preprocessing results.
- `modules/privacy.py`: optional masking of emails and phone numbers before analysis/storage.
- `modules/urgency_detection.py`: context-aware urgency phrase detection and conservative deadline phrase extraction.
- `modules/action_extraction.py`: potential action extraction with evidence, deadline association, and uncertainty metadata.
- `modules/database.py`: SQLite schema, CRUD operations, corrections, review state, and export queries.
- `modules/email_service.py`: CSV validation, synthetic-data loading, record construction, and batch analysis orchestration.
- `modules/evaluation.py`: reproducible train/test split, pipeline training, held-out metrics, confusion matrices, and reports.
- `models/train_models.py`: CLI training entry point and persisted model artifacts.
- `models/email_classifier.py`: model loading, prediction, confidence, and transparent rule-based priority adjustment.
- `pages/*.py`: separate dashboard, analyze, inbox, review, evaluation, feedback, and settings views.
- `data/synthetic_emails.csv`: permitted synthetic training and demo data.
- `tests/`: automated unit and integration tests for preprocessing, privacy, classifiers, urgency, actions, database, CSV validation, and exports.

The app trains separate priority and intent classifiers using TF-IDF plus Multinomial Naive Bayes and Logistic Regression. A shared reproducible stratified split is used for model comparison when feasible. The active model is configurable and defaults to Logistic Regression. Rule-based adjustments are kept separate from the model output, with evidence and review flags stored alongside both.

## Design

### Style options considered

- **Quiet operations console** (0.07): dense, calm, data-first interface with restrained color coding.
- **Signal atlas** (0.05): editorial dashboard with bold labels and a visual urgency spectrum.
- **Soft glass utility** (0.03): lighter translucent cards and gentler status treatments.

### Chosen direction: Quiet operations console

- **Design Movement:** Scandinavian information design translated into a modern incident-response console.
- **Core Principles:** high signal-to-noise, explicit uncertainty, consistent status color, and fast scanning.
- **Color Philosophy:** warm off-white surfaces and deep ink create trust; indigo marks model intelligence; amber, coral, and mint are reserved for urgency, review, and positive confirmation.
- **Layout Paradigm:** a persistent left navigation rail plus an asymmetric main workspace: headline metrics, then a wide evidence table beside compact signal panels.
- **Signature Elements:** small uppercase section labels, thin indigo rules, and pill badges with compact confidence bars.
- **Interaction Philosophy:** every suggestion is inspectable and reversible; controls expose why a record was flagged rather than hiding logic behind a button.
- **Animation:** restrained 150–220ms fades and progress transitions only; no decorative motion, no auto-advancing content.
- **Typography System:** system sans stack for legibility with bold display scale for page titles; tabular numbers for metrics; code-like styling for evidence phrases.
- **Brand Essence:** a transparent triage desk for teams handling too much incoming email — calm, accountable, explainable.
- **Brand Voice:** concise and candid. Example lines: “Suggestions, not verdicts.” and “Review the signal before you act.”
- **Wordmark & Logo:** a compact two-line mark built from a square signal glyph and the wordmark “inbox / signal”.
- **Signature Brand Color:** signal indigo `#4F46E5`.

## Delivery

The project will include a complete README with local setup, training, startup, test, deployment, CSV, security, privacy, troubleshooting, and limitation notes. The Streamlit runtime listens on port 3000 for managed Preview compatibility, while local startup remains `streamlit run app.py`. A route manifest is included for the managed project contract. Generated model/database/cache artifacts are ignored by Git.
