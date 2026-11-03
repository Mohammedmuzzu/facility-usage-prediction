"""Serialization and timestamp helpers; no credentials or external services."""
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.config import TIMEZONE


def timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Timestamp must be valid and include a timezone offset")
    return result.tz_convert(TIMEZONE)


def load_bookings(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    for column in ("booking_timestamp", "usage_timestamp"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="raise").dt.tz_convert(TIMEZONE)
    return data


def save_json(path: Path, value: Any) -> None:
    def encode(obj: Any) -> Any:
        if isinstance(obj, (pd.Timestamp, Path)):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minutes_of_day(series: pd.Series) -> np.ndarray:
    return (series.dt.hour * 60 + series.dt.minute + series.dt.second / 60).to_numpy(dtype=float)


def circular_error(actual: Any, predicted: Any) -> np.ndarray:
    distance = np.abs(np.asarray(actual) - np.asarray(predicted)) % 1440
    return np.minimum(distance, 1440 - distance)
