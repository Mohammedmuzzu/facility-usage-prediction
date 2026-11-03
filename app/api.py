"""Fail-closed production serving; development mode is local synthetic review."""

from contextlib import asynccontextmanager
from functools import lru_cache
import logging
import threading
import time
import uuid
from typing import Annotated
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Path as ApiPath, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from src.config import ROOT
from src.predict import PredictionEngine
from src.artifacts import ArtifactError
from src.utils import load_bookings, timestamp
from src.service_controls import (
    ServiceSettings,
    RateLimiter,
    BodyLimitMiddleware,
    authenticate,
    authorize,
)

LOG = logging.getLogger("facility.api")
RESIDENT_PATTERN = r"^R-(?:[1-9][0-9]{3}|0[1-9][0-9]{2}|00[1-9][0-9]|000[1-9])$"


class CutoffRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prediction_cutoff: str = Field(min_length=10, max_length=64)

    @field_validator("prediction_cutoff")
    @classmethod
    def aware_cutoff(cls, value: str) -> str:
        timestamp(value)
        return value


class PredictionRequest(CutoffRequest):
    resident_reference: str = Field(pattern=RESIDENT_PATTERN, max_length=6)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resident_reference: str
    prediction_cutoff: str
    predicted_facility: str
    predicted_usage_date: str
    predicted_usage_time: str
    predicted_usage_timestamp: str
    recommended_notification_timestamp: str
    cold_start: bool
    confidence_note: str
    notification_note: str
    model_dataset_sha256: str
    release_id: str
    history_count: int
    prediction_strategy: str
    component_routes: dict[str, str]
    validation_error_guides: dict[str, float]
    prediction_policy: str
    warnings: list[str]


@lru_cache(maxsize=1)
def get_engine() -> PredictionEngine:
    return PredictionEngine(
        ROOT / "models/model_bundle.joblib",
        load_bookings(ROOT / "data/facility_bookings.csv"),
    )


def create_app(
    settings: ServiceSettings | None = None, engine_provider=None
) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    provider = engine_provider or (lambda: get_engine())
    limiter = RateLimiter(settings.rate_db, settings.requests_per_minute)
    capacity = threading.BoundedSemaphore(settings.max_concurrency)
    loading, telemetry_lock = threading.Lock(), threading.Lock()
    counters = {
        "requests": 0,
        "errors": 0,
        "inference_requests": 0,
        "total_latency_seconds": 0.0,
    }
    engine_holder = {}

    def ready_engine():
        try:
            with loading:
                if "engine" not in engine_holder:
                    engine_holder["engine"] = provider()
            engine = engine_holder["engine"]
            if settings.environment == "production":
                age = (
                    pd.Timestamp.now(tz="UTC") - engine.latest_booking
                ).total_seconds() / 3600
                if age < -1 / 12 or age > settings.max_data_age_hours:
                    raise ValueError("History freshness policy failed")
            return engine
        except Exception as exc:
            LOG.warning("model_not_ready error_type=%s", type(exc).__name__)
            raise HTTPException(
                503, "Model or history snapshot unavailable; operator action required"
            ) from exc

    @asynccontextmanager
    async def lifespan(application):
        try:
            ready_engine()
        except HTTPException:
            pass
        yield

    application = FastAPI(
        title="Facility Usage Prediction",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.environment == "development" else None,
    )
    application.add_middleware(BodyLimitMiddleware, maximum=settings.max_body_bytes)

    @application.middleware("http")
    async def observe(request: Request, call_next):
        request_id, started = uuid.uuid4().hex, time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOG.error("request_failed request_id=%s", request_id)
            response = JSONResponse(
                {"detail": "Internal service error"}, status_code=500
            )
        elapsed = time.perf_counter() - started
        with telemetry_lock:
            counters["requests"] += 1
            counters["errors"] += int(response.status_code >= 400)
            counters["total_latency_seconds"] += elapsed
        response.headers.update(
            {
                "X-Request-ID": request_id,
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            }
        )
        LOG.info(
            "request_completed request_id=%s status=%d elapsed_ms=%.3f",
            request_id,
            response.status_code,
            elapsed * 1000,
        )
        return response

    def principal(x_api_key: Annotated[str | None, Header()] = None):
        return authenticate(settings, x_api_key)

    @application.get("/health")
    def health():
        return {"status": "ok", "purpose": "liveness"}

    @application.get("/ready")
    def readiness():
        return {"status": "ready", "release_id": ready_engine().release_id}

    @application.get("/model-info")
    def model_info(identity=Depends(principal)):
        authorize(identity)
        metadata = ready_engine().bundle["metadata"]
        return {
            key: metadata[key]
            for key in [
                "model_family",
                "ready_at",
                "seed",
                "dataset_sha256",
                "split",
                "dependencies",
                "routes",
            ]
        }

    @application.get("/metrics")
    def metrics(identity=Depends(principal)):
        authorize(identity)
        with telemetry_lock:
            return dict(counters)

    def infer(resident: str, cutoff: str, identity: dict):
        authorize(identity, resident)
        limiter.check(identity["key_sha256"])
        if settings.environment == "production" and timestamp(
            cutoff
        ) > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5):
            raise HTTPException(422, "Prediction cutoff must not be in the future")
        if not capacity.acquire(blocking=False):
            raise HTTPException(
                503, "Inference capacity reached", headers={"Retry-After": "1"}
            )
        try:
            engine = ready_engine()
            with telemetry_lock:
                counters["inference_requests"] += 1
            try:
                return engine.predict(resident, cutoff)
            except ArtifactError as exc:
                raise HTTPException(503, "Model compatibility check failed") from exc
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, "Invalid cutoff or inference input") from exc
        finally:
            capacity.release()

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest, identity=Depends(principal)):
        return infer(request.resident_reference, request.prediction_cutoff, identity)

    @application.post(
        "/predict/{resident_reference}", response_model=PredictionResponse
    )
    def predict_resident(
        resident_reference: Annotated[str, ApiPath(pattern=RESIDENT_PATTERN)],
        request: CutoffRequest,
        identity=Depends(principal),
    ):
        return infer(resident_reference, request.prediction_cutoff, identity)

    return application


app = create_app()
