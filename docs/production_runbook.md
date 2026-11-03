# Production operations and release conditions

## Deployment boundary

This is a hardened single-community service. It is not a provisioned enterprise environment or an approved real-world reminder policy. The PDF prototype is runnable; business release stays blocked until E01–E04 in the audit are supplied. Do not use the synthetic success rates as a production SLA.

Provision a reviewed TLS/SSO reverse proxy, secret manager, private network, log/metrics collector, organization retention/access rules and approved quality/latency/error thresholds. Gateway limits must cover header/body read deadlines, slow clients, maximum connections and multi-host quotas. SQLite quotas cover only workers sharing one local volume; in-memory telemetry is per process and resets on restart. Export/scrape aggregate counters externally. Do not enable Uvicorn access logs containing resident paths.

## Keys and scopes

Create keys with scripts/create_api_key.py. The generated raw value is displayed once; store it in the secret manager. Only SHA-256 is written to .secrets/auth.json (mode 0600). A principal has explicit resident references or admin=true. No anonymous wildcard resident scope exists. X-API-Key is required whenever an auth file is configured. Production requires an auth file; missing or invalid configuration fails closed. Rotating a principal replaces its hash; subsequent requests use the updated file. Mount a readable secret for UID 10001, never commit it or bake it into an image. Docker Compose secrets can retain host file ownership/mode, so verify readability explicitly for your secret driver; adjust controlled group ownership or provision the secret using your deployment platform rather than making it world-readable.

## Configuration

| Variable | Default / purpose |
|---|---|
| FACILITY_ENV | development locally; production in container |
| FACILITY_AUTH_FILE | Required production scoped-hash JSON |
| FACILITY_PROJECT_ROOT | Artifact/history root, default repository/app root |
| FACILITY_RATE_DB | .cache/api_rate.sqlite3 locally; /tmp/facility-rate.sqlite3 in container |
| FACILITY_RATE_LIMIT | 120 requests per principal per fixed minute |
| FACILITY_MAX_BODY_BYTES | 16,384 actual bytes, including streamed payloads |
| FACILITY_MAX_CONCURRENCY | 4 inferences per worker; configure gateway aggregate limit |
| FACILITY_MAX_DATA_AGE_HOURS | 24 hours; production readiness rejects stale/future snapshots |
| FACILITY_DASHBOARD_TRUSTED_PROXY | true only after deploying a private authenticated proxy |

Production /docs and OpenAPI are disabled. /health is liveness. /ready loads and verifies the immutable release/history, checks freshness, and returns 503 on failure. /metrics and /model-info require an admin. Monitoring logs only server-generated request IDs, status and duration. Avoid attaching resident labels, request bodies, credentials or internal exception strings to external telemetry.

## Data ingestion

Export a complete five-column snapshot using the documented timezone-aware schema. Run ingest_bookings.py with --input and --expected-current-sha256. It validates data, preserves the full trained history and currently accepted observations, rejects corrections/deletions/backfills and uses an optimistic hash plus a file lock before atomic replacement. Append-only additions must be strictly later than the previous snapshot maximum. Real providers must map cancellations/status/capacity explicitly before integration; the present schema is scheduled bookings only.

Restart serving processes after ingestion to load the new immutable in-memory history. The model remains frozen. Historical corrections require retraining and a new release. Benchmark evaluation requires the exact original training snapshot hash and intentionally rejects a live appended file. Keep training/evaluation snapshots separate from live deployment inputs in a real system.

## Model release and rollback

Train into a controlled workspace using the hashed lock, run tests, inspect validation decisions, benchmark/confirmation reports, data coverage, class errors, drift and business gates. Review the trusted release/image digest. models/releases/<id> holds the complete immutable package and manifest; current.json is the only serving pointer. Activation validates the release before switching. Existing workers retain the old loaded release; restart/drain them to change versions.

Use python scripts/manage_release.py RELEASE_ID for rollback/activation. Never edit an immutable directory or replace hashes to bypass failure. Keep the previous compatible release and its matching data snapshot/image. Source/runtime mismatches require the corresponding old image as well as old weights. Retain only approved releases according to organizational retention policy; remove older directories only after confirming no active pointer/process references them. This repository does not pretend a local directory is an authenticated cloud model registry.

## Container verification

Build with docker build -t facility-api:2 . and deploy the provided compose.yaml behind the private gateway. The image uses a pinned official Python digest, exact hashed runtime dependencies, UID 10001, disabled access logs, finite Uvicorn connection/backlog/shutdown controls and a readiness health check. Compose adds loopback publication, read-only filesystem, tmpfs, dropped capabilities, no-new-privileges, CPU/memory bounds and a mounted secret. CI separately smoke-tests development mode with synthetic data. Production rejects the bundled historical dataset as stale; that is expected until valid recent data is integrated.

## Incidents and monitoring

- 401/403: verify key rotation, resident scope and admin access; never log supplied keys.
- 413/429: inspect gateway byte limits or principal quota; quotas are fixed-minute, so boundary bursts are possible.
- 503 readiness: inspect trusted artifact hashes, dependency/source compatibility, full history preservation, recent data and future-clock errors. Do not replace a 503 with a false 200.
- 503 capacity: reduce incoming concurrency or scale within tested memory/CPU limits; use gateway admission control for multiple workers/hosts.
- Elevated drift/errors: inspect PSI and labelled per-class/time metrics, data age and source schema. PSI is diagnostic; cumulative activity naturally changes. Retrain only through documented development selection and evaluate prospectively.
- Unexpected 500: correlate request ID with sanitized operator events, reproduce in a protected staging environment, roll back the reviewed image/release if necessary.

Health, metrics and observed latency are engineering signals. Alert thresholds, uptime SLOs, retention, recovery objectives and notification policy require owner decisions and cannot be invented from a synthetic assignment.
