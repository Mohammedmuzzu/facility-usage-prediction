from dataclasses import replace
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.api import create_app
from src.artifacts import (
    ArtifactError,
    activate_release,
    load_bundle,
    history_contract,
    validate_history,
    resolve_release,
)
from src.feature_engineering import FeatureBuilder
from src.dataset_builder import build_examples
from src.data_validation import validate_bookings
from src.predict import PredictionEngine, assemble_predictions
from src.service_controls import ServiceSettings, RateLimiter
from src.temporal_split import validate_splits
from src.utils import load_bookings
from src.monitoring import reference_profile, drift_report


@pytest.mark.parametrize(
    "change",
    [
        {"residents": 10000},
        {"seed": -1},
        {"months": 0},
        {"exploration": float("nan")},
        {"sparse_fraction": 1.1},
        {"popularity": (1, 2)},
        {"start": "2025-01-15"},
        {"bookings": True},
    ],
)
def test_bad_generator_configuration(config, change):
    with pytest.raises(ValueError):
        replace(config, **change)


@pytest.mark.parametrize("value", [123, "", "R-0000", None])
def test_malformed_resident_is_rejected(bookings, value):
    changed = bookings.copy()
    changed["resident_reference"] = changed.resident_reference.astype(object)
    changed.loc[0, "resident_reference"] = value
    with pytest.raises(ValueError):
        validate_bookings(changed)


def test_csv_rejects_naive_and_corrupt_timestamps(bookings, tmp_path):
    path = tmp_path / "bad.csv"
    changed = bookings.copy()
    changed["booking_timestamp"] = changed.booking_timestamp.dt.tz_localize(None)
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="timezone"):
        load_bookings(path)
    changed = bookings.copy()
    changed["booking_timestamp"] = changed.booking_timestamp.astype(str)
    changed.loc[0, "booking_timestamp"] = "invalid+05:30"
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_bookings(path)


def test_history_is_defensively_copied(bookings):
    row = bookings.iloc[0]
    builder = FeatureBuilder(bookings)
    expected = builder.build(row.resident_reference, row.booking_timestamp)
    history = builder.history(row.resident_reference, row.booking_timestamp)
    history.loc[:, "facility_reference"] = "Tennis Court"
    assert builder.build(row.resident_reference, row.booking_timestamp) == expected


def test_ties_are_invariant_to_input_order(bookings):
    sample = bookings.iloc[:30].copy()
    sample.loc[:, "resident_reference"] = "R-0001"
    sample.loc[1, "booking_timestamp"] = sample.loc[0, "booking_timestamp"]
    left = build_examples(sample)
    right = build_examples(sample.iloc[::-1])
    for a, b in zip(left, right):
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize(
    "case", ["overlap", "duplicate", "out_of_bounds", "non_integer", "chronology"]
)
def test_split_guards(examples, case):
    _, y, original, _ = examples
    splits = {name: values.copy() for name, values in original.items()}
    if case == "overlap":
        splits["test"][0] = splits["train"][0]
    elif case == "duplicate":
        splits["train"][1] = splits["train"][0]
    elif case == "out_of_bounds":
        splits["test"][0] = len(y)
    elif case == "non_integer":
        splits["train"] = splits["train"].astype(float)
    else:
        splits["train"], splits["test"] = splits["test"], splits["train"]
    with pytest.raises(ValueError):
        validate_splits(y, splits)


@pytest.mark.parametrize(
    "case", ["nan", "infinity", "wrong_length", "facility", "scalar"]
)
def test_prediction_component_validation(case):
    raw = {"facility": ["Gym"], "days": [2.0], "minutes": [420.0], "lead_hours": [2.0]}
    if case == "nan":
        raw["days"] = [float("nan")]
    elif case == "infinity":
        raw["minutes"] = [float("inf")]
    elif case == "wrong_length":
        raw["days"] = [2, 3]
    elif case == "facility":
        raw["facility"] = ["Invalid"]
    else:
        raw["lead_hours"] = 2.0
    with pytest.raises(ValueError):
        assemble_predictions(
            pd.Series([pd.Timestamp("2026-06-01T10:00:00+05:30")]), raw
        )


def test_empty_prediction_batch():
    result = assemble_predictions(
        pd.Series([], dtype="object"),
        {key: [] for key in ["facility", "days", "minutes", "lead_hours"]},
    )
    assert result.empty


def test_manifest_checked_before_unpickling(trained, tmp_path, monkeypatch):
    import shutil
    import src.artifacts as artifacts

    _, root = trained
    release = resolve_release(root / "model_bundle.joblib").parent
    copy = tmp_path / "release"
    shutil.copytree(release, copy)
    (copy / "model_bundle.joblib").write_bytes(b"corrupt")
    called = []
    monkeypatch.setattr(artifacts.joblib, "load", lambda *args: called.append(True))
    with pytest.raises(ArtifactError, match="integrity"):
        load_bundle(copy / "model_bundle.joblib", follow_pointer=False)
    assert not called


def test_rollback_and_bad_pointer_guard(trained, tmp_path):
    import shutil

    _, original = trained
    root = tmp_path / "models"
    shutil.copytree(original, root)
    active = json.loads((root / "current.json").read_text())["release_id"]
    activate_release(root, active)
    with pytest.raises(ArtifactError):
        activate_release(root, "../escape")
    with pytest.raises(ArtifactError):
        activate_release(root, "a" * 32)
    assert json.loads((root / "current.json").read_text())["release_id"] == active


def test_append_only_history_contract(bookings):
    contract = history_contract(bookings)
    validate_history(bookings, contract)
    with pytest.raises(ArtifactError):
        validate_history(bookings.iloc[:-1], contract)
    changed = bookings.copy()
    changed.loc[0, "facility_reference"] = "Tennis Court"
    with pytest.raises(ArtifactError):
        validate_history(changed, contract)
    added = bookings.iloc[[-1]].copy()
    added["booking_id"] = "B-new"
    added["booking_timestamp"] += pd.Timedelta(days=2)
    added["usage_timestamp"] += pd.Timedelta(days=2)
    validate_history(pd.concat([bookings, added]), contract)
    added["booking_timestamp"] = bookings.booking_timestamp.min()
    with pytest.raises(ArtifactError):
        validate_history(pd.concat([bookings, added]), contract)


@pytest.fixture
def secured(bookings, trained, tmp_path):
    _, root = trained
    engine = PredictionEngine(root / "model_bundle.joblib", bookings)
    keys = {"resident": "resident-test-key", "admin": "admin-test-key"}
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps(
            {
                "principals": [
                    {
                        "id": name,
                        "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                        "residents": ["R-0001"] if name == "resident" else [],
                        "admin": name == "admin",
                    }
                    for name, key in keys.items()
                ]
            }
        )
    )
    settings = ServiceSettings(auth_file=str(auth), rate_db=str(tmp_path / "quota.db"))
    return settings, engine, keys


def test_auth_scopes_strict_inputs_and_telemetry(secured):
    settings, engine, keys = secured
    with TestClient(create_app(settings, lambda: engine)) as client:
        cutoff = (engine.latest_booking + pd.Timedelta(days=1)).isoformat()
        payload = {"resident_reference": "R-0001", "prediction_cutoff": cutoff}
        assert client.post("/predict", json=payload).status_code == 401
        resident = {"X-API-Key": keys["resident"]}
        response = client.post("/predict", json=payload, headers=resident)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert len(response.headers["x-request-id"]) == 32
        assert (
            client.post(
                "/predict",
                json={**payload, "resident_reference": "R-0002"},
                headers=resident,
            ).status_code
            == 403
        )
        assert client.get("/model-info", headers=resident).status_code == 403
        assert (
            client.post(
                "/predict", json={**payload, "unrecognized": True}, headers=resident
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/predict/R-0000", json={"prediction_cutoff": cutoff}, headers=resident
            ).status_code
            == 422
        )
        admin = {"X-API-Key": keys["admin"]}
        assert client.get("/metrics", headers=admin).json()["inference_requests"] == 1
        assert client.get("/ready").status_code == 200


def test_fail_closed_production_and_stale_readiness(secured):
    with pytest.raises(ValueError):
        ServiceSettings(environment="production")
    settings, engine, _ = secured
    settings = replace(settings, environment="production", max_data_age_hours=0.001)
    with TestClient(create_app(settings, lambda: engine)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/docs").status_code == 404


def test_corrupt_model_readiness_is_sanitized(tmp_path):
    def broken():
        raise ArtifactError("secret resident path /private/test")

    settings = ServiceSettings(rate_db=str(tmp_path / "quota.db"))
    with TestClient(create_app(settings, broken)) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        assert "secret" not in response.text and "/private" not in response.text


def test_size_limit_including_chunked_payload(secured):
    settings, engine, _ = secured
    with TestClient(
        create_app(replace(settings, max_body_bytes=64), lambda: engine)
    ) as client:
        response = client.post("/predict", content=(b"a" * 40 for _ in range(3)))
        assert response.status_code == 413


def test_rate_quota_shared_and_atomic(tmp_path):
    first = RateLimiter(str(tmp_path / "quota.db"), 5)
    second = RateLimiter(str(tmp_path / "quota.db"), 5)

    def attempt(i):
        try:
            (first if i % 2 else second).check("same-principal", now=120)
            return 200
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(20)))
    assert results.count(200) == 5 and results.count(429) == 15
    first.check("same-principal", now=180)


def test_monitoring_references_are_finite(examples):
    X, _, splits, _ = examples
    reference = reference_profile(X.iloc[splits["train"]])
    report = drift_report(reference, X.iloc[splits["test"]])
    assert report["sample_count"] == len(splits["test"])
    assert all(np.isfinite(value["psi"]) for value in report["features"].values())


def test_concurrency_backpressure_and_recovery(secured, monkeypatch):
    settings, engine, keys = secured
    original = engine.predict
    started, release = threading.Event(), threading.Event()

    def slow(*args):
        started.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(engine, "predict", slow)
    with TestClient(
        create_app(replace(settings, max_concurrency=1), lambda: engine)
    ) as client:
        payload = {
            "resident_reference": "R-0001",
            "prediction_cutoff": (
                engine.latest_booking + pd.Timedelta(days=1)
            ).isoformat(),
        }
        headers = {"X-API-Key": keys["admin"]}
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(client.post, "/predict", json=payload, headers=headers)
            assert started.wait(10)
            try:
                assert (
                    client.post("/predict", json=payload, headers=headers).status_code
                    == 503
                )
            finally:
                release.set()
            assert future.result().status_code == 200
        assert client.post("/predict", json=payload, headers=headers).status_code == 200


def test_runtime_failure_has_no_pii_or_secret_in_response_or_logs(
    secured, monkeypatch, caplog
):
    settings, engine, keys = secured

    def broken(*args):
        raise RuntimeError("private location resident R-0001 secret-value")

    monkeypatch.setattr(engine, "predict", broken)
    with TestClient(create_app(settings, lambda: engine)) as client:
        response = client.post(
            "/predict",
            headers={"X-API-Key": keys["admin"]},
            json={
                "resident_reference": "R-0001",
                "prediction_cutoff": (
                    engine.latest_booking + pd.Timedelta(days=1)
                ).isoformat(),
            },
        )
        assert response.status_code == 500
        assert "private location" not in response.text + caplog.text
        assert "secret-value" not in response.text + caplog.text
        assert "R-0001" not in caplog.text


def test_credential_rotation_and_invalid_file_fail_closed(secured):
    settings, engine, keys = secured
    with TestClient(create_app(settings, lambda: engine)) as client:
        headers = {"X-API-Key": keys["admin"]}
        assert client.get("/metrics", headers=headers).status_code == 200
        location = Path(settings.auth_file)
        saved = location.read_text()
        location.write_text('{"principals": []}')
        assert client.get("/metrics", headers=headers).status_code == 503
        location.write_text(saved)
        assert client.get("/metrics", headers=headers).status_code == 200


def test_changed_history_rejected_at_engine_load(bookings, trained):
    _, root = trained
    changed = bookings.copy()
    changed.loc[0, "facility_reference"] = (
        "Gym" if changed.loc[0, "facility_reference"] != "Gym" else "Tennis Court"
    )
    with pytest.raises(ArtifactError, match="snapshot"):
        PredictionEngine(root / "model_bundle.joblib", changed)


def test_incompatible_runtime_fails_before_deserialization(
    trained, tmp_path, monkeypatch
):
    import shutil
    import src.artifacts as artifacts

    _, root = trained
    copy = tmp_path / "release"
    shutil.copytree(resolve_release(root / "model_bundle.joblib").parent, copy)
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["dependencies"]["numpy"] = "0.0.0"
    (copy / "manifest.json").write_text(json.dumps(manifest))
    called = []
    monkeypatch.setattr(artifacts.joblib, "load", lambda *args: called.append(True))
    with pytest.raises(ArtifactError, match="runtime"):
        load_bundle(copy / "model_bundle.joblib", follow_pointer=False)
    assert not called
