# Facility Usage Prediction — Technical Documentation, v2

## Executive summary

The PDF asks for a working synthetic dataset, leakage-safe facility/date/hour/notification prediction, unseen chronological evaluation, technical documentation and an actual-versus-predicted review. These deliverables are implemented. The architecture audit adds integrity-checked immutable releases, strict input contracts, scoped serving controls, drift diagnostics, operational packaging and failure-path testing. [The audit register](architecture_audit.md) lists every identified gap; [the resolution register](audit_resolution.md) distinguishes verified fixes from external release requirements.

This is a conditional next-observed-booking model. It does not estimate whether any booking will occur. Notification creation time is a booking-readiness proxy, not experimentally optimized delivery. Engineering readiness cannot substitute for prospective business validation.

## Business and data understanding

One synthetic residential community contains 500 residents, six facilities, 12,500 observations and 18 calendar months, seed 42. Timezone is Asia/Kolkata. [The dictionary](data_dictionary.md) defines the exact five-field schema. Scheduled usage is known at booking creation; attendance is not observed.

The generator uses unequal resident activity, latent fitness/swimming/sports/social/occasional/mixed preferences, weighted facility popularity, noisy recurrent and irregular sessions, weekend/weekday effects, early gym/evening sports/afternoon social usage, seasonal demand, resident-specific lead variation, late arrivals, inactivity, exploration and preference changes. All parameters are validated and configurable. Latent personas/change points never enter exported data or features. This is a hand-specified simulation, not a statistical fit to actual community logs. It has no capacity, cancellation or attendance labels.

Validation checks required fields/order, identifier types/ranges, unique booking IDs, missing data, explicit timezone parsing, known facilities, positive lead <=720 hours and creation chronology. Profiles record sparsity, class imbalance, activity and lead quantiles, missing classes and complete date ranges. Missing facility classes are reported; training requires at least two observed classes. Tiny or degenerate chronological partitions fail explicitly.

![Popularity](../outputs/figures/facility_distribution.png)
![Activity](../outputs/figures/resident_activity.png)
![Leads](../outputs/figures/lead_time_distribution.png)
![Hours](../outputs/figures/usage_hour_distribution.png)

## Sequential samples and feature availability

Within each resident, order by creation timestamp then booking ID. Each distinct creation cutoff predicts the next strictly later event. All simultaneous cutoff bookings are observable; a simultaneous next target uses the smallest booking ID and records its multiplicity. The same FeatureBuilder handles training, API and dashboard. Every aggregate is computed after filtering creation <= cutoff. Already-created bookings may contain scheduled usage later than the cutoff; those scheduled values are observable, not attendance outcomes. Returned histories are defensive copies.

Features cover counts and activity in 7/30/90 days, diversity, creation interval mean/median/last gap/CV, elapsed time, frequency, lead summaries, most recent/mode facility, facility-specific counts/lead/circular time, lifetime and exponentially recent Laplace-smoothed preferences, cutoff weekday/hour/month/season/weekend, previous/preferred usage weekday/hour and circular means including seconds/nanoseconds. [Feature definitions](feature_documentation.md) document fixed defaults. Residents with <2 observations use train-only global timing fallbacks; the cold_start diagnostic identifies <3. IDs, future counts, next facility/creation/usage and hidden personas are excluded from X.

Cutoffs in the standard task occur at booking creation. Later arbitrary cutoffs are supported with explicit warnings and corrected remaining-interval baselines, but their policy performance is not prospectively established. Cold-start evaluation is a separate first-observation cohort; it is not silently mixed into the previous-event task.

## Chronological evaluation

Global next-booking creation timestamps establish earliest 70%, next 15%, final 15%. Ties stay together. Validation/test rows whose cutoff precedes the relevant model-ready boundary are purged. All estimators and baselines share partitions. Training enforces unique, disjoint, in-bounds integer indices, strictly later targets and chronology before fitting. Encoders learn only training categories. Models fit training only and use validation for candidate/route selection. Early stopping is disabled to avoid a random internal split. No train+validation refit occurs.

Sequentially observed validation/test history may update resident features after creation, while fitted parameters remain frozen. Scheduled usage/lead labels are available at creation, so attendance outcome maturity is not assumed. [Coverage](../outputs/coverage_report.json) reconciles every booking, initial/non-target tied observations, supervised examples, purges, sparse-user effects and final censored prediction cutoffs. Last observed bookings can be targets while their own next cutoffs remain censored.

The original holdout was inspected in v1; v2 calls it a reused regression benchmark. Original metrics are preserved in outputs/audit/before. A predeclared independent simulation seed 20261103 is evaluated after the revised implementation is frozen, with validation-only selection. That experiment is independent synthetic confirmation, not real-world validation. Three expanding development-only windows add calendar-regime coverage without accessing the original final test.

## Modelling and serving selection

Five CPU HistGradientBoosting estimators implement four outputs: a multiclass facility classifier with square-root inverse-frequency weights; absolute-error elapsed-day regression; squared-error sine/cosine usage-time regressions; and absolute-error booking-lead regression. A fixed two-configuration grid varies leaves, iterations and regularization. Main facility candidates select validation macro F1; days/leads use MAE; circular pairs use joint shortest-arc validation MAE. Preprocessing is persisted with each estimator.

ML, resident-history and global baselines are independently evaluated. Global defaults are train-target mode/medians. The resident baseline uses past facility mode, remaining median creation interval plus historical lead, preferred hour and median lead. A serving candidate selects facility by validation macro F1, days by validation elapsed MAE, time by circular MAE and lead by end-to-end proxy MAE using selected usage context. Thus the service can retain a heuristic when ML adds no validation value. It never chooses a route from test results and does not promise a win on every test metric.

Day outputs are bounded to 0–365 elapsed days. The implied date and circular time are composed into a future usage timestamp, preserving time of day while enforcing cutoff/horizon bounds. Lead is bounded to 0.25–720 hours. Reminder proxy is usage minus lead, constrained after cutoff and at least 15 minutes before usage. Scalar, mismatched, nonfinite and invalid-facility components fail explicitly; empty batches are supported. Circular resultants near zero fall back to historical time. API response times retain nanoseconds in ISO strings; human-readable clock labels are displays, never equality inputs.

Logistic regression, forests, CatBoost, Markov transitions, collaborative filtering and sequence networks were considered. Lightweight nonlinear tabular models suit this modest data size and four heterogeneous targets. Native CatBoost categorical handling is a valid future comparator. These alternatives are not falsely presented as benchmarked. No deep-learning/LLM infrastructure is introduced.

## Metrics, uncertainty and diagnosis

Facility: accuracy, macro/weighted F1, per-class precision/recall/support and confusion matrix. Days: raw elapsed MAE/median AE, weekday accuracy, within 1/2/3 elapsed days and calendar-date MAE. Time: shortest circular MAE/median AE and within 30/60/120 minutes. Notification: full proxy timestamp MAE/median AE and within 1/3/6 hours, plus independent lead error. Date/time errors propagate into notification proxy error.

Exact matching means facility, date, clock-hour bucket and full reminder timestamp equality. Tolerance means facility exact, calendar date within one day, circular time within 60 minutes and reminder within three hours. Both 4/4 rates and averages out of four are reported separately, without rounding raw predictions. The workbook has all retained serving-candidate test rows, five formatted/filterable/frozen sheets, all variants' metrics, explanations and coverage/route provenance.

Resident-block percentile bootstrap intervals preserve within-resident dependence. Validation p90 error guides are empirical descriptions, not calibrated probabilities or prediction-interval guarantees. Error slices include sparse/frequent histories, facility rarity, calendar-quarter transitions and a history-only preference-shift proxy. Missing slice support is explicitly reported. Train-only feature histogram references supply PSI diagnostics; >0.2 is an investigation heuristic, not automatic proof of failure, especially for cumulative activity features.

[Generated results](evaluation_report.md), [full report](../outputs/evaluation_report.json), [development diagnostics](../outputs/robustness_report.json) and [independent confirmation](../outputs/audit/confirmation_report.json) contain actual numbers. No performance improvement, seasonal robustness or notification effectiveness is invented.

## Architecture and operational boundaries

Models publish to a staged immutable release, validate hashes/runtime/schema and atomically switch current.json. Flat models are compatibility exports; serving follows the pointer. Runtime dependencies, source hashes, feature names, parameters, dataset hash, history record fingerprints and selection audit are persisted. The history contract permits later append-only bookings but rejects changed/deleted originals and backfills. Ingestion additionally protects all currently accepted observations using a file lock, expected hash and atomic write. Old processes retain their pinned release; restart after activation/rollback.

Manifest verification occurs before Joblib loading. A manifest is a corruption check, not an attacker-authenticating signature. Protect the reviewed image/release and deployment write access. The API has scoped hashed keys, admin-only diagnostics, strict request/response schemas, request byte/quota/concurrency limits, sanitized errors, request IDs and aggregate metrics. Production fails without credentials and returns 503 for missing/corrupt/incompatible/stale artifacts/data. Liveness remains separate. SQLite quotas coordinate processes on one host/volume; distributed quotas and TLS/SSO/slow-client limits require a gateway.

The container pins its official base digest and runtime hashes, removes package installers, runs without root and uses readiness probes. Compose bounds resources and privilege. The dashboard is an admin review surface and requires a private authenticated proxy in production; a form gate alone cannot protect Streamlit media endpoints. No external notification delivery, cloud deployment, vendor connector or organizational SSO has been fabricated. See [operations](production_runbook.md) and [external release conditions](limitations.md).

## References and decisions informed

- [Scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): train-only preprocessing and shared inference transformations.
- [Scikit-learn persistence](https://scikit-learn.org/stable/model_persistence.html): trusted artifact boundary and exact environment compatibility.
- [Scikit-learn temporal cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split): chronological evaluation; custom timestamp windows for irregular events.
- [Histogram gradient boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting): CPU tabular models/loss choices.
- [Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html): per-class/macro metrics rather than accuracy alone.
- [CatBoost categorical handling](https://catboost.ai/docs/en/features/categorical-features): native categorical alternative; not used in this submission.
- [Ji et al., recommender evaluation leakage](https://arxiv.org/abs/2010.11060): global chronology instead of per-user splits that learn from later events elsewhere.
- [Camargo et al., generative event logs](https://arxiv.org/abs/2009.03567): explicit simulation assumptions and realism limits; no implementation copied.
- [FastAPI credential comparison](https://fastapi.tiangolo.com/advanced/security/http-basic-auth/): constant-time secret comparison; this service uses hashed scoped API keys, not HTTP Basic.
- [Docker build practices](https://docs.docker.com/build/building/best-practices/): pinned base, limited build context and non-root runtime.
