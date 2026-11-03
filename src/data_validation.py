"""Fail closed on malformed input; report, rather than hide, class imbalance."""
import pandas as pd
from src.config import FACILITIES, TIMEZONE, GenerationConfig

SCHEMA = ["booking_id", "resident_reference", "facility_reference", "booking_timestamp", "usage_timestamp"]


def validate_bookings(data: pd.DataFrame, config: GenerationConfig | None = None) -> dict:
    errors = []
    if list(data.columns) != SCHEMA:
        raise ValueError(f"Expected exactly these ordered columns: {SCHEMA}")
    if data.empty or data.isna().any().any():
        raise ValueError("Dataset is empty or contains missing values")
    if data.booking_id.duplicated().any():
        errors.append("duplicate booking IDs")
    if not data.resident_reference.str.fullmatch(r"R-\d{4}").all():
        errors.append("invalid resident reference")
    if config and not data.resident_reference.isin([f"R-{i:04d}" for i in range(1, config.residents + 1)]).all():
        errors.append("resident outside configured population")
    if not data.facility_reference.isin(FACILITIES).all():
        errors.append("invalid facility reference")
    for column in ["booking_timestamp", "usage_timestamp"]:
        if not isinstance(data[column].dtype, pd.DatetimeTZDtype) or str(data[column].dt.tz) != TIMEZONE:
            errors.append(f"{column} requires {TIMEZONE} timezone-aware datetimes")
    if errors:
        raise ValueError("; ".join(errors))
    lead = (data.usage_timestamp - data.booking_timestamp).dt.total_seconds() / 3600
    if (lead <= 0).any() or (lead > 720.001).any():
        errors.append("booking must precede usage by at most 30 days")
    if not data.booking_timestamp.is_monotonic_increasing:
        errors.append("bookings must be chronologically ordered")
    if config and (len(data) != config.bookings or data.resident_reference.nunique() != config.residents):
        errors.append("configured population/record counts differ")
    if errors:
        raise ValueError("; ".join(errors))
    counts = data.resident_reference.value_counts()
    distribution = data.facility_reference.value_counts().reindex(FACILITIES, fill_value=0)
    return {
        "validation_passed": True, "residents": int(len(counts)), "facilities": len(FACILITIES), "bookings": len(data),
        "timezone": TIMEZONE, "booking_range": [str(data.booking_timestamp.min()), str(data.booking_timestamp.max())],
        "usage_range": [str(data.usage_timestamp.min()), str(data.usage_timestamp.max())],
        "facility_distribution": distribution.to_dict(), "resident_booking_frequency": counts.describe().to_dict(),
        "lead_time_hours": lead.describe(percentiles=[.1, .5, .9, .99]).to_dict(),
        "residents_with_at_most_5_bookings": int((counts <= 5).sum()),
        "imbalance_ratio": float(distribution.max() / max(1, distribution.min())),
        "generation_config": config.to_dict() if config else None,
        "seed": config.seed if config else None,
    }
