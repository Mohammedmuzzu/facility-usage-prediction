# Architecture and requirements audit

Baseline: commit `3b26d6a`, PDF sections 1–4 and the supplied end-to-end assignment. The PDF requires a reproducible working prototype; production operations are an additional request. This register distinguishes code defects from missing business evidence. No finite test suite proves every possible scenario.

| ID | Gap in the original implementation | Required correction |
|---|---|---|
| A01 | Inference loads Joblib before integrity/compatibility validation | Verify manifest, format, runtime, feature schema and history before serving |
| A02 | Model and CSV can silently disagree at inference | Bind release to an immutable history snapshot; validate append-only updates |
| A03 | Multi-file training writes can expose partial releases | Stage immutable releases, atomically activate, support rollback |
| A04 | Pickled supervised cache can be stale or executable | Remove executable feature cache; rebuild from validated input |
| A05 | CSV parser silently interprets naive timestamps as UTC | Require explicit timezone, validate schema/types and reject malformed inputs |
| A06 | Generator configuration is incompletely bounded | Validate population, fractions, positive weights, date/calendar limits |
| A07 | Equal creation timestamps rely on input row order | Deterministic booking-ID tie policy and explicit tie coverage |
| A08 | Initial and censored bookings, and temporal purges, are invisible in review coverage | Record every exclusion and population/slice coverage |
| A09 | Split inputs are trusted by training | Enforce disjoint indices, bounds, chronology and model-ready cutoffs before fitting |
| A10 | Previous booking gap is absent; circular features discard seconds | Add precise historical gap and circular features with consistent defaults |
| A11 | Returned resident history may alias mutable internal data | Defensive copies and invariance tests |
| A12 | Late-cutoff heuristic adds elapsed time twice | Subtract elapsed time from resident next-event interval estimates |
| A13 | Batch prediction trusts shapes, nonfinite values and facility labels | Validate all components, empty batches, bounds and time ordering |
| A14 | Event-driven training is presented as arbitrary-cutoff assurance | Return cutoff-policy/data-age warnings and explicit readiness/freshness contracts |
| A15 | ML is served even when a heuristic is better | Validation-only component routing; preserve independently evaluated ML and both baselines |
| A16 | Weak timing precision lacks operational release criteria | Explicit technical/business gates and no invented success thresholds |
| A17 | One holdout cannot establish seasonal/general temporal robustness | Development-only expanding windows and scenario coverage report |
| A18 | Existing holdout has already been inspected | Label reused benchmark; predeclare an independent confirmation seed before revised fitting |
| A19 | Metrics lack sampling uncertainty and evaluated cold-start cohort | Resident-block bootstrap and first-observation cohort with chronology controls |
| A20 | No drift or stale-data monitoring | Train-only reference distributions, drift diagnostics and freshness checks |
| A21 | Unauthenticated resident predictions enable identifier-based disclosure | Hashed API credentials, resident scopes, admin-only diagnostics; production fail-closed |
| A22 | No request size/rate/concurrency bounds | Enforce byte limit, bounded inference, process-safe SQLite quota and strict request schemas |
| A23 | Liveness and readiness are conflated | Load/validate on startup; 503 readiness for missing/corrupt/stale artifacts |
| A24 | Errors expose internals; no request telemetry | Sanitized errors, request IDs, aggregate metrics and PII-free structured logs |
| A25 | Dashboard can expose all resident data when deployed | Production authentication gate; document admin-only scope |
| A26 | No deployment or rollback/runbook package | Non-root container, read-only deployment recipe, release/rollback and incident procedures |
| A27 | CI resolves transitive dependencies afresh and lacks security/static checks | Lock-based installs, lint, dependency audit, container smoke and uploaded evidence |
| A28 | Workbook lacks overall exclusion evidence and explicit route provenance | Preserve five core sheets; add coverage/routes and consistent raw precision |
| A29 | Source/configuration changes are not bound to artifacts | Save code hashes, full estimator parameters and release manifest |
| A30 | Tests omit corruption, isolation, scoped authorization and overload paths | Regression tests for every implemented boundary and failure path |
| E01 | No real resident logs, capacity/availability, cancellations or no-show labels | External data integration and acceptance tests required; never invent these fields |
| E02 | No notification engagement/conversion experiment | Optimal notification effectiveness cannot be established; retain proxy warning |
| E03 | No organization SSO, TLS endpoint, secrets manager, retention or SLO decisions | Supply deployment hooks/runbook; organization must provision and approve these |
| E04 | No probability-of-booking labels or prospective real-world validation | Retain conditional task; explicit business release gate blocks effectiveness claims |

Implementation status, executed evidence and remaining external dependencies are recorded in `audit_resolution.md` after verification. Original scores are preserved in `outputs/audit/before/`. The revised original test is a regression benchmark. Independent confirmation uses seed **20261103**, fixed now, 500 residents / 12,500 bookings / 18 months, evaluated only after the implementation and selection criteria are frozen. Confirmation outcomes must not drive retuning.
