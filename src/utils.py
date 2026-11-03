"""Serialization and timestamp helpers; no credentials or external services."""

import hashlib
import json
import os
import tempfile
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
    data = pd.read_csv(
        path,
        dtype={
            "booking_id": "string",
            "resident_reference": "string",
            "facility_reference": "string",
        },
    )
    from src.data_validation import SCHEMA, validate_bookings

    if list(data.columns) != SCHEMA:
        raise ValueError(f"Expected exactly these ordered columns: {SCHEMA}")
    for column in ("booking_timestamp", "usage_timestamp"):
        if (
            not data[column]
            .astype(str)
            .str.fullmatch(r".*(?:Z|[+-]\d{2}:?\d{2})")
            .all()
        ):
            raise ValueError(f"{column} must include an explicit timezone offset")
        data[column] = pd.to_datetime(
            data[column], utc=True, errors="raise", format="ISO8601"
        ).dt.tz_convert(TIMEZONE)
    validate_bookings(data)
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
    atomic_text(
        path, json.dumps(value, indent=2, default=encode, allow_nan=False) + "\n"
    )


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".staging-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def frame_hash(data: pd.DataFrame) -> str:
    canonical = data.sort_values(
        ["booking_timestamp", "booking_id"], kind="stable"
    ).reset_index(drop=True)
    return hashlib.sha256(canonical.to_csv(index=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minutes_of_day(series: pd.Series) -> np.ndarray:
    return (
        series.dt.hour * 60
        + series.dt.minute
        + series.dt.second / 60
        + series.dt.microsecond / 60_000_000
        + series.dt.nanosecond / 60_000_000_000
    ).to_numpy(dtype=float)


def circular_error(actual: Any, predicted: Any) -> np.ndarray:
    distance = np.abs(np.asarray(actual) - np.asarray(predicted)) % 1440
    return np.minimum(distance, 1440 - distance)
