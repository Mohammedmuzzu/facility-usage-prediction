"""Run: uvicorn app.api:app --reload. Models load lazily, never train."""
import json
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import ROOT
from src.predict import PredictionEngine
from src.utils import load_bookings

app = FastAPI(title="Facility Usage Prediction", version="1.0.0")


class PredictionRequest(BaseModel):
    resident_reference: str = Field(min_length=1, max_length=100)
    prediction_cutoff: str = Field(description="Timezone-aware ISO 8601 timestamp")


class CutoffRequest(BaseModel):
    prediction_cutoff: str


@lru_cache(maxsize=1)
def get_engine() -> PredictionEngine:
    return PredictionEngine(ROOT / "models/model_bundle.joblib", load_bookings(ROOT / "data/facility_bookings.csv"))


@app.get("/health")
def health() -> dict:
    ready = (ROOT / "models/model_bundle.joblib").exists() and (ROOT / "data/facility_bookings.csv").exists()
    return {"status": "ok" if ready else "artifacts_missing", "artifacts_ready": ready}


@app.get("/model-info")
def model_info() -> dict:
    if not (ROOT / "models/model_metadata.json").exists():
        raise HTTPException(503, "Run the training pipeline first")
    metadata = json.loads((ROOT / "models/model_metadata.json").read_text())
    return {key: metadata[key] for key in ["model_family", "ready_at", "seed", "dataset_sha256", "split", "dependencies"]}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        return get_engine().predict(request.resident_reference, request.prediction_cutoff)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "Run the pipeline to create model and dataset artifacts") from exc


@app.post("/predict/{resident_reference}")
def predict_resident(resident_reference: str, request: CutoffRequest) -> dict:
    return predict(PredictionRequest(resident_reference=resident_reference, prediction_cutoff=request.prediction_cutoff))
