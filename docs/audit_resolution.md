# Architecture audit resolution and executed evidence

Baseline reviewed: 3b26d6a. [Original findings](architecture_audit.md) and [source traceability](requirements_traceability.md) cover the PDF and detailed prompt. The register distinguishes implemented engineering corrections from business/deployment conditions that cannot be fabricated.

| Finding | Resolution in this revision | Evidence |
|---|---|---|
| A01 | Verify every artifact digest/runtime/schema before trusted deserialization | test_manifest_checked_before_unpickling, test_incompatible_runtime_fails_before_deserialization |
| A02 | Trained history fingerprints and validated append-only additions | test_append_only_history_contract, test_changed_history_rejected_at_engine_load |
| A03 | Staged immutable releases; atomic pointer; validated rollback | test_rollback_and_bad_pointer_guard; artifacts.py |
| A04 | Removed executable supervised cache; rebuild from validated CSV | workflow.supervised; clean full pipeline |
| A05 | Strict types/IDs/schema and explicit timezone parsing | malformed resident, naive/corrupt CSV and eight invalid-data tests |
| A06 | Bounded and finite configuration including seed/date/population | test_bad_generator_configuration and extreme-regime tests |
| A07 | Stable booking-ID tie policy with multiplicity | test_ties_are_invariant_to_input_order; coverage report |
| A08 | Complete record reconciliation, sparse purge effects and censored cutoff counts | outputs/record_coverage.csv and coverage_report.json |
| A09 | Guard partition bounds/overlap/chronology before fit | test_split_guards; final-test poisoning isolation |
| A10 | Previous booking gap and precise circular historical time | feature documentation and cutoff feature tests |
| A11 | Defensive history copies | test_history_is_defensively_copied |
| A12 | Resident remaining-interval baseline subtracts elapsed time | baselines.py; explicit arbitrary-cutoff warning |
| A13 | Shape, finite, facility, empty-batch and horizon validation | component tests; calendar/year/leap/midnight edge tests |
| A14 | Explicit event-conditional policy, stale-history/cutoff warnings; live freshness readiness | API response schema; stale production readiness test |
| A15 | Validation-only ML/heuristic routing for each component | saved routes/losses, independent metrics, poisoned-test invariance |
| A16 | Persisted technical/business gate distinction | evaluation_report.json release_gates; no invented effectiveness threshold |
| A17 | Three expanding development-only windows | robustness_report.json; no-final-holdout-access test |
| A18 | Preserved v1 metrics, labelled reused benchmark, predeclared independent seed | outputs/audit/before and confirmation_report.json |
| A19 | Resident-cluster bootstrap plus genuine first-observation cohort | full report uncertainty; robustness report cohort counts |
| A20 | Train-only reference histograms/PSI and production data-age checks | drift/reference tests; stale readiness test |
| A21 | Hash-only keys, resident scopes, admin diagnostics, rotation and fail-closed config | authorization, scope, rotation and missing-config tests |
| A22 | Actual streamed-byte bounds, atomic shared quota, inference backpressure | size/stream/quota/concurrency/recovery tests |
| A23 | Startup artifact validation, separate liveness/readiness and sanitized 503 | corrupt-model and production readiness tests |
| A24 | Request IDs, no-store headers, sanitized logs/errors and aggregate metrics | test_runtime_failure_has_no_pii_or_secret_in_response_or_logs |
| A25 | Admin dashboard gate before data load; required private production proxy | AppTest development smoke; deployment runbook; E03 external gateway remains required |
| A26 | Pinned non-root image, read-only deployment, lifecycle/incident runbook | Dockerfile, compose.yaml, container smoke evidence; no cloud provision claim |
| A27 | Cross-platform hashed locks, lint/format, dependency audit and CI evidence | outputs/audit/runtime_dependency_audit.json; locked clean environment; GitHub CI |
| A28 | Five core sheets, all variants, precise values and route/coverage provenance | workbook row/style/value tests and generated Dataset Insights |
| A29 | Full estimator settings, input history contract, source hashes and manifest | model_metadata.json and immutable release manifest |
| A30 | Expanded regression and failure-path coverage | 73 passing tests; outputs/audit/test_results.xml |

## Executed checks

- 73 tests passed in the working environment and 73 in a fresh directory/environment.
- Fresh source was copied without data, models, outputs or feature caches; the full pipeline generated all artifacts itself. Dataset, serving-test CSV, metrics JSON and split JSON reproduced byte-identically. See clean_environment_verification.json.
- Lint and formatter checks pass. Hash locks cover Linux and macOS, including Linux-only Watchdog; the container build caught and drove correction of the initially host-only lock.
- Dependency scanning originally reported advisories in six installed packages, including bootstrap/build tools. Affected versions were upgraded. Follow-up installed-environment and final-runtime-lock scans found no known Python-package advisories in their reported scope. This is not an operating-system image scan or a guarantee of absence of vulnerabilities.
- Dashboard overview, tabs, resident prediction and result/charts review passed Streamlit AppTest.
- Three development-only temporal evaluations include 199, 323 and 18 quarter-transition observations respectively. The original final holdout still has none, and the report states that limitation. Genuine first-observation cold-start cohort n=3 is underpowered and separately labelled.
- Original reused regression benchmark n=1,413. Independent predeclared seed 20261103 confirmation n=1,428, evaluated after code/selection criteria freeze. No confirmation-driven retuning was performed.
- Container image built with pinned base/runtime hashes. Its non-root/read-only startup and inference checks are recorded in container_smoke.json after successful verification. Build failures uncovered Linux lock coverage and ownership of owner-only atomic artifacts; both were corrected before publication.

The suite retains upstream deprecation warnings involving Starlette TestClient/HTTPX and Matplotlib/Pyparsing. These do not represent failed checks and are not hidden by global warning suppression.

## Remaining external release conditions

E01 real booking/availability/status data; E02 notification engagement and controlled conversion evidence; E03 organization gateway/SSO/TLS/secrets/retention/SLO provisioning; E04 prospective real-world task/occurrence validation. These are **open external requirements**, with hooks and runbook supplied. They are not marked solved. The business production-approval flag remains false. No finite scenario matrix or synthetic benchmark establishes universal coverage.

Current serving-candidate benchmark: facility accuracy 48.48%, macro F1 0.3866, elapsed-day MAE 5.27, circular-time MAE 181.68 minutes, notification-proxy MAE 128.32 hours, tolerance 4/4 0.64%. Independent confirmation: facility accuracy 44.12%, day MAE 5.78, circular-time MAE 179.78 minutes, notification-proxy MAE 134.64 hours, tolerance 4/4 0.98%. These weak timing precision results justify the blocked business gate rather than a production-effectiveness claim. Full unrounded values, every baseline and subgroup support are in generated JSON/Excel artifacts.
