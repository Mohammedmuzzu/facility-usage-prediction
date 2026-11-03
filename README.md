# Facility Usage Prediction System

CPU-only next-booking modelling and a hardened single-community inference service. Includes synthetic data, four independently evaluated outputs, trained models, temporal evaluation, a five-sheet Excel review, dashboard, authenticated API, container, audit register, and automated checks.

**Engineering controls are implemented. Real-world business effectiveness is not established.** The release gate remains blocked on real booking/availability data, organization-approved service and quality thresholds, prospective validation, and notification engagement experiments. Predictions are conditional on another booking occurring. Reminder timing is a historical booking-time proxy.

## Audit and evidence

- [All 30 engineering findings and four external requirements](docs/architecture_audit.md)
- [Resolution register and verification](docs/audit_resolution.md)
- [PDF and assignment requirement traceability](docs/requirements_traceability.md)
- [Actual benchmark results](docs/evaluation_report.md), [complete metrics](outputs/evaluation_metrics.json), [coverage](outputs/coverage_report.json)
- [Independent confirmation](outputs/audit/confirmation_report.json), [development backtests](outputs/robustness_report.json)
- [Excel review](outputs/facility_prediction_review.xlsx), [every original observation and exclusion](outputs/record_coverage.csv)
- [Production runbook](docs/production_runbook.md), [technical documentation](docs/technical_documentation.md), [limitations](docs/limitations.md)

The original dataset has 12,500 bookings, 500 residents, six facilities and 18 calendar months (January 2025–June 2026), seed 42, Asia/Kolkata. The previously inspected holdout is explicitly a **reused regression benchmark**. A separately predeclared seed 20261103 supplies independent synthetic confirmation, never a claim about real residents.

## Run from a clean environment

Python 3.11 is tested. Use the exact hashed lock; model loading rejects incompatible NumPy/Pandas/scikit-learn/Joblib versions.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python scripts/run_pipeline.py
python -m pytest -q
ruff check src app scripts tests
```

The complete pipeline generates and validates data, rebuilds cutoff-safe features, fits models on training data, selects on validation, evaluates the retained chronological benchmark, and generates CSV/JSON/Excel/charts. Seeds, partitions, source hashes, full estimator parameters, preprocessing, validation decisions and data contracts are recorded. Test outputs never select models. There is no executable supervised-feature cache.

Individual stages and diagnostics:

```bash
python scripts/generate_data.py --seed 42 --residents 500 --bookings 12500 --months 18
python scripts/train_models.py
python scripts/evaluate_models.py
python scripts/generate_predictions.py
python scripts/generate_predictions.py --resident R-0001 --cutoff '2026-07-01T12:00:00+05:30'
python scripts/run_robustness.py
python scripts/explain_model.py
```

Evaluation/prediction never fit models. Data changes require an explicit retrain for evaluation. `run_robustness.py` fits separate development-only windows; none reaches the original final test. `run_confirmation.py` reproduces the predeclared independent experiment; use it as a confirmation run, never as a tuning loop.

## Models, evaluation and data coverage

Five histogram gradient-boosting estimators implement facility classification, elapsed-day regression, two sine/cosine regressions for circular usage time, and advance-lead regression. Categorical preprocessing fits only training rows. Inverse-frequency weighting and macro F1 explicitly address class imbalance. Resident-history/global heuristics are independently reported. Serving chooses each component from ML/resident/global using validation alone; its selected routes are recorded and fixed before testing.

Every feature sees only resident bookings created at or before its cutoff. Targets use the next strictly later creation event, with booking-ID tie breaking. Global target-creation timestamps establish 70/15/15 partitions; crossing cutoffs are purged so the fitted/selected model exists by the prediction cutoff. The coverage report reconciles all 12,500 observations and shows sparse-user exclusion effects. Already-observed holdout history can update features; weights remain frozen.

`outputs/test_predictions.csv` and the workbook show the **validation-selected serving candidate**. `outputs/ml_test_predictions.csv` preserves independent raw-ML results. Both baselines are separate files. Reports include per-class metrics, timing/notification errors, exact and tolerance 4/4, subgroup counts, resident-cluster bootstrap intervals, train-reference drift diagnostics, and release gates.

Exact means facility, calendar date, clock-hour bucket and full notification timestamp equality. Tolerances mean facility exact, date within one calendar day, time within 60 circular minutes and notification within three hours. No prediction is rounded to manufacture equality. Excel displays local time; nanosecond fidelity remains in CSV.

## Local API and dashboard

Local development uses synthetic data and is intentionally accessible without a key. Bind it to loopback:

```bash
uvicorn app.api:app --host 127.0.0.1 --no-access-log
streamlit run app/streamlit_app.py --server.address=127.0.0.1
```

Routes: `GET /health` (liveness), `GET /ready` (validated readiness), `GET /model-info` and `/metrics` (admin), `POST /predict`, `POST /predict/{resident_reference}`. OpenAPI is available in development only.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"resident_reference":"R-0001","prediction_cutoff":"2026-07-01T12:00:00+05:30"}'
```

[Example saved-model response](outputs/example_prediction.json). Responses expose the model release, route provenance, history count, empirical validation error guides and cutoff/freshness warnings. Error guides are not calibrated probability or coverage guarantees. Bad/naive timestamps, invalid residents, unknown fields and unsupported historical cutoffs return 422. Overload returns 503; quota excess returns 429.

## Production service controls

```bash
python scripts/create_api_key.py --principal operator --admin
python scripts/create_api_key.py --principal resident-1 --resident R-0001
FACILITY_ENV=production FACILITY_AUTH_FILE=.secrets/auth.json uvicorn app.api:app --host 127.0.0.1 --no-access-log
```

Provision generated keys in your secret manager; only hashes are stored in the gitignored owner-only file. Production requires credentials, validates model/data integrity before deserialization/serving, enforces resident scopes, freshness, bounded request bytes, per-principal SQLite quotas and concurrent inference limits. Keys rotate on the next request. Predictions are read-only and send no notifications. Logs contain request IDs/status/latency, not resident identifiers, bodies or credentials.

The supplied historical dataset is deliberately stale relative to a live deployment. Production readiness returns **503** until a recent valid append-only snapshot is supplied. Do not loosen freshness to misrepresent this as live data. Production dashboard additionally requires a private authenticated reverse proxy and `FACILITY_DASHBOARD_TRUSTED_PROXY=true`; it is an admin review tool.

```bash
docker build -t facility-api:2 .
docker compose up --build
```

The image pins the official Python base digest, uses a hashed runtime lock, runs as UID 10001, removes package installers, and has a readiness health check. Compose binds loopback, uses read-only root filesystem/tmpfs, drops capabilities and bounds CPU/memory. TLS, organization SSO, slow-client/body timeouts, multi-host quotas, secret provisioning and external monitoring belong at your deployment gateway; see the runbook. No cloud infrastructure has been provisioned.

## Artifact lifecycle

`models/releases/<id>/` is immutable. A manifest verifies every artifact before trusted Joblib deserialization; `models/current.json` changes atomically only after a complete release passes validation. Flat model files remain assignment-compatible exports. The manifest detects corruption; it does not authenticate an attacker-replaced manifest. Trust must come from a reviewed image/release digest and restricted deployment access.

```bash
python scripts/manage_release.py RELEASE_ID
```

Restart serving processes after activation/rollback. Existing processes pin their loaded release. Append-only ingestion requires a complete validated snapshot and expected current CSV hash:

```bash
python scripts/ingest_bookings.py --input /path/to/complete_snapshot.csv --expected-current-sha256 CURRENT_HASH
```

It rejects removed/changed observations and backfills, uses a process-safe file lock and atomically replaces the CSV. Historical corrections require explicit retraining. This is a CSV adapter with a strict contract, not a fabricated integration with a real booking vendor.

## Repository

`src/`: data, features, chronology, models, inference, releases, metrics, monitoring and service controls. `app/`: API/dashboard. `scripts/`: executable stages and operations. `tests/`: regression and failure-path coverage. `data/`: synthetic CSV/profile. `models/`: active immutable release and compatibility exports. `outputs/`: predictions/workbooks/figures/evidence. `docs/`: audit, methodology and operations. `notebooks/`: optional EDA companion. CI installs the hashed lock, checks formatting/lint/dependencies, runs tests and full pipeline, and builds/smoke-tests the container.

The supplied confidential assignment documents, GitHub tokens, service keys and development caches are excluded. See [data dictionary](docs/data_dictionary.md), [features](docs/feature_documentation.md) and [repository file list](docs/repository_structure.txt).
