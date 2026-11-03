"""One next strictly future booking per unique resident creation cutoff."""
import pandas as pd
from src.feature_engineering import FeatureBuilder


def build_examples(bookings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    builder = FeatureBuilder(bookings)
    features, targets = [], []
    for resident, history in builder.histories.items():
        # Tied creation events are all visible together; no arbitrary order within a tie.
        for cutoff in history.booking_timestamp.drop_duplicates().iloc[:-1]:
            future = history[history.booking_timestamp > cutoff].iloc[0]
            features.append(builder.build(resident, cutoff))
            targets.append({"booking_id": future.booking_id, "resident_reference": resident,
                            "prediction_cutoff": cutoff, "target_booking_timestamp": future.booking_timestamp,
                            "target_usage_timestamp": future.usage_timestamp, "target_facility": future.facility_reference,
                            "days_until_usage": (future.usage_timestamp - cutoff).total_seconds() / 86400,
                            "lead_hours": (future.usage_timestamp - future.booking_timestamp).total_seconds() / 3600})
    if not targets:
        raise ValueError("At least one resident needs two distinct booking creation timestamps")
    X, y = pd.DataFrame(features), pd.DataFrame(targets)
    order = y.sort_values(["target_booking_timestamp", "booking_id"], kind="stable").index
    return X.loc[order].reset_index(drop=True), y.loc[order].reset_index(drop=True)
