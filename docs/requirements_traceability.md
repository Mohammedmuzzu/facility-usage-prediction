# PDF and assignment traceability

| Source requirement | Implementation | Verification / limits |
|---|---|---|
| PDF 1, 3.1 business/schema | data_generator, config, data_validation; exact five business/technical fields | Generator reproducibility, schema/type/date/lead tests; real activity not established |
| PDF 3.1 meaningful behaviour | Persona-weighted popularity, timing, lead, sparse/late/irregular activity, seasons and drift | Default profile and extreme regime tests; latent labels excluded |
| PDF 3.2 four predictions | train, baselines, predict, validation-selected routes | Independent raw ML, resident/global, serving-candidate metrics |
| PDF 3.2 leakage-safe pipeline | FeatureBuilder, dataset_builder, temporal_split | Append/modify future invariance, test-poison isolation, chronology, train-only encoders |
| PDF 3.3 chronological holdout | Global 70/15/15 plus model-ready purge | Saved exact boundaries; all observations reconciled; old holdout reuse disclosed |
| PDF 3.3 suitable/overall metrics | evaluate, monitoring, reporting | Facility per-class/F1; elapsed/circular/proxy MAE; exact/tolerance 4/4; block bootstrap |
| PDF 3.3 error analysis | Activity, facility rarity, historical shift and seasonal slices | Support counts and no-quarter-transition caveat; development-only windows |
| PDF 3.4/3.5 readable comparison | Five-sheet workbook and dashboard; CSV per strategy | Every retained test row, raw error values, freeze/filter/format tests |
| PDF 4 runnable submission | README, source, data, models, locks, scripts, docs, CI | Fresh environment, full pipeline, tests, container smoke; private GitHub repository |
| Prompt 3 generation quality | Bounded configuration, timezone strictness, profile | All core validations, malformed and extreme inputs |
| Prompt 4 sequential task | Next strictly later event, stable tied-target rule | Deterministic reordered-tie test, full exclusion coverage |
| Prompt 5 features | Counts/rolling/gaps/diversity/facility preferences/lead/calendar/circular | Historical cutoff tests and documented defaults; IDs/personas absent from X |
| Prompt 6 models | Weighted HGB, elapsed-day, sine/cosine, lead proxy | CPU models, independent files, baselines and raw metrics |
| Prompt 7 evaluation/reproducibility | Fixed candidates/seeds, train-only transforms, validation routes, immutable releases | Split guard tests; metadata/source/data hashes; locked environment |
| Prompt 8 overall/error analysis | Explicit exact/tolerance thresholds and slice diagnostics | Raw-value equality test, bootstrap/drift/coverage reports |
| Prompt 9 mandatory workbook | Prediction Review, Performance Summary, Detailed Metrics, Error Analysis, Dataset Insights | All five sheet names/rows/styles/value precision verified; route/coverage in insights |
| Prompt 10 optional dashboard | Overview, resident prediction/history, filters, metrics/charts/download | AppTest; production admin gate plus required private authenticated proxy |
| Prompt 11 optional API | Health/readiness/info/metrics/two prediction routes | Saved-artifact prediction, scoped keys, validation/corruption/overload/quota tests |
| Prompt 12–15 structure/tests/docs/run | Modular source, scripts, all artifacts and operations documentation | Ruff, hashed-lock clean run, pytest, CI container build/smoke |
| Prompt 16 research | Primary references in technical documentation | Scikit-learn chronology/persistence/metrics, CatBoost, event-log/leakage research |
| Prompt 17–19 honesty/completion | Original results preserved; regression versus confirmation distinguished | No manufactured accuracy, no test-selected routes, blocked real-world effectiveness gate |

All implemented controls have finite, explicit contracts. The PDF does not provide actual availability/capacity, cancellations/no-shows, notifications/conversions, organization identity/TLS/retention/SLOs or booking-occurrence labels. These remain E01–E04 in the audit; they cannot honestly be solved by inventing data or claiming universal scenario coverage.
