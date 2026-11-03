# Facility Usage Prediction System

A working, CPU-only next-booking prediction system for one synthetic residential community. It predicts facility, scheduled usage date, usage time, and a **booking-time proxy** for a reminder. The repository includes generated data, trained models, a complete chronological test review, Excel workbook, plots, API, dashboard, tests, and technical documentation.

**500 residents · 6 facilities · 12,500 bookings · January 2025–June 2026 · seed 42 · Asia/Kolkata**

The problem is conditional on another booking occurring. It does not estimate whether a resident will book at all. The notification output does not establish an effective notification policy.

## Start here

- [Technical documentation](docs/technical_documentation.md)
- [Actual results and baseline comparisons](docs/evaluation_report.md)
- [Prediction review workbook](outputs/facility_prediction_review.xlsx) — all 1,413 retained final-test records, five formatted sheets
- [Actual versus predicted CSV](outputs/test_predictions.csv)
- [All evaluation metrics](outputs/evaluation_metrics.json) and [error analysis](outputs/evaluation_report.json)
- [Dataset and generation metadata](data/dataset_metadata.json)
- [Known limitations](docs/limitations.md)

Facility ML improves macro F1 but has lower accuracy than the resident-mode baseline. The timing regressions improve average absolute errors, while the resident baseline achieves more matches at the strict business tolerances. Exact 4/4 is zero. These results are reported as measured; this is an engineering prototype, not a deployment claim.

## Reproduce from a clean environment

Python **3.11** is the verified runtime. Python 3.12 may also work; other versions/platforms can produce small floating-point differences. No GPU or external account is needed to run the project.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/run_pipeline.py
python -m pytest -q
```

`requirements-lock.txt` pins the full tested environment, including development dependencies. `requirements.txt` pins the direct runtime dependencies; `requirements-dev.txt` adds test tools. The full workflow regenerates the same default synthetic CSV, explicitly retrains on the training partition, selects configurations on validation, and evaluates on the retained final test partition. Typical execution is under a few minutes on a laptop. Random seeds and candidate configurations are fixed; final test results are never used for selection.

Stages can also be run separately:

```bash
python scripts/generate_data.py
python scripts/train_models.py
python scripts/evaluate_models.py
python scripts/generate_predictions.py
```

The last two commands load saved artifacts and never retrain. `generate_predictions.py` regenerates the workbook, CSVs, metrics, and charts from the frozen artifacts. A hash mismatch between model and data fails evaluation. The ignored `.cache/` directory accelerates repeated feature construction; its key includes dataset and feature/split source hashes. It can be deleted safely.

```bash
python scripts/generate_data.py --seed 42 --residents 500 --bookings 12500 --months 18
python scripts/generate_predictions.py --resident R-0001 --cutoff '2026-07-01T12:00:00+05:30'
```

Changing generation parameters requires explicit retraining. More behavioural controls are centralized in `src/config.py`.

## Dashboard and API

```bash
streamlit run app/streamlit_app.py
uvicorn app.api:app --reload
```

The dashboard provides community overview, resident history at a selected cutoff, saved-model predictions, result filters, workbook download, baseline metrics, and charts. It does not show uncalibrated classifier scores as probabilities.

API routes: `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/{resident_reference}`. OpenAPI: `http://127.0.0.1:8000/docs`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"resident_reference":"R-0001","prediction_cutoff":"2026-07-01T12:00:00+05:30"}'
```

A real saved-model response is in [outputs/example_prediction.json](outputs/example_prediction.json). It includes predicted facility, usage date/time, notification proxy timestamp, cold-start indicator, dataset hash, and limitations. Unseen residents receive global/train-only fallbacks. Timezone-naive cutoffs and cutoffs before model selection are rejected with HTTP 422. This local prototype has no authentication layer; production access control and real-data integration are future work.

## How the evaluation works

For each resident, a previous booking creation timestamp becomes the cutoff for the next strictly later booking. All feature inputs satisfy `booking_timestamp <= cutoff`. Models never receive the next facility, next booking time, next usage time, future preferences, or synthetic personas.

Targets define globally chronological 70% / 15% / 15% partitions. Validation and test examples crossing their model-ready boundary are purged, leaving 8,400 training, 1,409 validation, and 1,413 test examples; 778 crossing-cutoff examples are excluded. This protects infrequent residents from being scored using a model fitted or selected after their cutoff. All four outputs share these partitions. Previously observed holdout history may update resident features; models remain frozen.

Five HistGradientBoosting estimators implement four components: a weighted facility classifier, elapsed-day regressor, sine/cosine usage-time regressors, and lead-time regressor. Training-only one-hot encoding is persisted with every estimator. The facility objective explicitly addresses imbalance; model selection uses validation macro F1. Time pairs are selected by circular validation MAE. Days and leads use absolute-error loss. A resident-history and global baseline are evaluated on exactly the same test rows and constraints.

Exact matching uses facility, calendar date, clock-hour bucket, and full notification timestamp equality. Business tolerances are facility exact, usage date within one calendar day, usage time within 60 circular minutes, and notification within three elapsed hours. Raw values are not rounded to manufacture matches.

## Repository map

```text
data/           Synthetic CSV and dataset metadata
src/            Generation, validation, cutoff features, splitting, training,
                frozen inference, baselines, metrics, reporting, workflow
models/         Model bundle, five independent estimators, experiment metadata
outputs/        Workbook, all test/baseline predictions, JSON metrics, figures
app/            FastAPI service and Streamlit reviewer dashboard
tests/          Reproducibility, input checks, leakage, isolation, models,
                circular arithmetic, inference, API, metrics, Excel review
scripts/        Runnable workflow stages and validation diagnostics
docs/           Technical report, architecture, dictionaries, results, limits
notebooks/      Lightweight companion EDA; not required by the pipeline
.github/        CI executes tests and the full pipeline on Python 3.11
```

Test coverage includes appending/modifying future records without changing historical features; poisoning final-test features and outcomes without changing trained models or validation selection; train-only preprocessing; cold starts; model serialization; raw exact/tolerance calculations; and workbook row count, precision, sheet names, filters, and formatting. See [verification record](docs/verification.md) for executed checks.

Only load trusted local Joblib artifacts. The supplied assignment documents and GitHub credentials are not part of this repository.
