# Executed verification

Verified on 3 November 2026 using Python 3.11.13 on macOS arm64. These are completed checks, not a planned checklist.

| Check | Result |
|---|---|
| Default domain-aware generation and quality checks | Passed: 12,500 records, 500 residents, six facilities |
| Full generation → fitting → holdout evaluation → Excel/chart workflow | Passed |
| Automated tests | 24 passed |
| Fresh environment installed from full pinned lock, copied source, no cached features/models/data | Passed |
| Full pipeline in fresh directory and environment | Passed |
| Fresh-environment automated tests | 24 passed |
| Byte equality between working and clean runs | Identical CSV dataset, test-prediction CSV, metrics JSON and split JSON |
| Excel inspection | Five expected sheets and 1,413 final-test data rows; filters, frozen headers, conditional formatting and stored error values verified |
| Streamlit AppTest | Overview, four tabs, resident prediction button, review table and evaluation charts render without exceptions |
| Running FastAPI process | `/health`, `/model-info`, `/predict` returned HTTP 200 |
| Running Streamlit process | `/_stcore/health` returned HTTP 200 and `ok` |
| Validation-only feature importance | Generated without fitting or final-test access |

Tests cover generator seed reproducibility/sensitivity, population/schema/timestamps, eight malformed-input cases, ordered histories, cold starts, cutoff filtering, future append/modification invariance, strictly later targets, final-test poisoning isolation, temporal purging, training-only encoder vocabularies, model serialization/loading, circular midnight distances, notification arithmetic/constraints, inference availability, API validation, exact-versus-tolerance semantics, and Excel row/value/style checks.

The fresh run used a separate virtual environment populated from `requirements-lock.txt` and a copied source tree excluding data, models, outputs, caches and the development virtual environment. It generated all artifacts itself. [Machine-readable comparison](../outputs/reproducibility_verification.json) records the exact equality checks. XLSX ZIP metadata and model binary file hashes are not claimed to be byte-identical across runs; evaluated predictions and metrics are.

The executed unit/integration suite reports 15 upstream deprecation warnings involving the pinned Starlette/AnyIO and Matplotlib/Pyparsing combinations. There are no failed tests. Dependency upgrades would require revalidation; warnings are not suppressed to conceal failures.

The optional companion notebook is exploratory source, not the execution path used to create submission artifacts. GitHub CI configuration is included; a remote CI result is only claimed if separately observed. No real notification is delivered by any check.

Reproduce:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/run_pipeline.py
python -m pytest -q
python scripts/explain_model.py
```

JUnit output is included in `outputs/test_results.xml`; live service response evidence is in `outputs/service_smoke_test.json`. The current environment requires sandbox permission for local port binding; the documented application commands themselves run successfully.
