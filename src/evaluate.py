"""Raw-value metrics and explicitly separate exact and tolerance matches."""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.config import FACILITIES
from src.utils import circular_error, minutes_of_day


def review_predictions(
    y: pd.DataFrame, predicted: pd.DataFrame, X: pd.DataFrame
) -> pd.DataFrame:
    if (
        y.empty
        or len(y) != len(predicted)
        or len(y) != len(X)
        or not y.index.equals(X.index)
    ):
        raise ValueError(
            "Evaluation requires nonempty, aligned targets, predictions and features"
        )
    review = pd.concat(
        [y.reset_index(drop=True), predicted.reset_index(drop=True)], axis=1
    )
    actual, pred = review.target_usage_timestamp, review.predicted_usage_timestamp
    review["facility_match"] = review.target_facility == review.predicted_facility
    review["usage_date_error_days"] = np.abs(
        (actual.dt.normalize() - pred.dt.normalize()).dt.days
    )
    review["usage_interval_error_days"] = (
        np.abs((actual - pred).dt.total_seconds()) / 86400
    )
    review["days_regression_error"] = np.abs(
        review.days_until_usage - review.predicted_days_until_usage
    )
    review["usage_time_error_minutes"] = circular_error(
        minutes_of_day(actual), minutes_of_day(pred)
    )
    review["notification_error_hours"] = (
        np.abs(
            (
                review.target_booking_timestamp
                - review.predicted_notification_timestamp
            ).dt.total_seconds()
        )
        / 3600
    )
    review["lead_model_error_hours"] = np.abs(
        review.lead_hours - review.predicted_model_lead_hours
    )
    review["date_exact"] = actual.dt.date == pred.dt.date
    review["hour_exact"] = actual.dt.hour == pred.dt.hour
    review["notification_exact"] = (
        review.target_booking_timestamp == review.predicted_notification_timestamp
    )
    review["date_tolerance"] = review.usage_date_error_days <= 1
    review["time_tolerance"] = review.usage_time_error_minutes <= 60
    review["notification_tolerance"] = review.notification_error_hours <= 3
    exact = ["facility_match", "date_exact", "hour_exact", "notification_exact"]
    tolerance = [
        "facility_match",
        "date_tolerance",
        "time_tolerance",
        "notification_tolerance",
    ]
    review["exact_matches_out_of_4"] = review[exact].sum(axis=1)
    review["tolerance_matches_out_of_4"] = review[tolerance].sum(axis=1)
    review["overall_result"] = np.where(
        review.exact_matches_out_of_4 == 4,
        "Correct",
        np.where(
            review.tolerance_matches_out_of_4 == 4, "Within tolerance", "Incorrect"
        ),
    )
    review["history_count"] = X.total_bookings.to_numpy()
    review["activity_group"] = pd.cut(
        review.history_count,
        [-1, 4, 19, np.inf],
        labels=["Sparse (0-4)", "Moderate (5-19)", "Frequent (20+)"],
    ).astype(str)
    review["season_transition"] = (
        review.prediction_cutoff.dt.quarter != actual.dt.quarter
    )
    # Drift proxy uses historical distributions alone; it never accesses latent synthetic personas.
    pref = X[[f"facility_{i}_preference" for i in range(6)]].to_numpy()
    recent = X[[f"facility_{i}_recent_preference" for i in range(6)]].to_numpy()
    review["history_preference_shift"] = np.abs(pref - recent).sum(axis=1) > 0.12
    return review


def metrics(review: pd.DataFrame) -> dict:
    date = review.days_regression_error
    time = review.usage_time_error_minutes
    notification = review.notification_error_hours
    return {
        "sample_count": len(review),
        "facility": {
            "accuracy": float(review.facility_match.mean()),
            "macro_f1": float(
                f1_score(
                    review.target_facility,
                    review.predicted_facility,
                    labels=list(FACILITIES),
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    review.target_facility,
                    review.predicted_facility,
                    labels=list(FACILITIES),
                    average="weighted",
                    zero_division=0,
                )
            ),
            "per_class": classification_report(
                review.target_facility,
                review.predicted_facility,
                labels=list(FACILITIES),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                review.target_facility,
                review.predicted_facility,
                labels=list(FACILITIES),
            ).tolist(),
            "class_order": list(FACILITIES),
        },
        "usage_day": {
            "mae_days": float(date.mean()),
            "median_ae_days": float(date.median()),
            "calendar_date_mae_days": float(review.usage_date_error_days.mean()),
            "weekday_accuracy": float(
                (
                    review.target_usage_timestamp.dt.day_of_week
                    == review.predicted_usage_timestamp.dt.day_of_week
                ).mean()
            ),
            **{f"within_{d}_days": float((date <= d).mean()) for d in [1, 2, 3]},
        },
        "usage_hour": {
            "circular_mae_minutes": float(time.mean()),
            "median_circular_ae_minutes": float(time.median()),
            **{f"within_{m}_minutes": float((time <= m).mean()) for m in [30, 60, 120]},
        },
        "notification": {
            "mae_hours": float(notification.mean()),
            "median_ae_hours": float(notification.median()),
            "standalone_lead_mae_hours": float(review.lead_model_error_hours.mean()),
            "standalone_lead_median_ae_hours": float(
                review.lead_model_error_hours.median()
            ),
            **{
                f"within_{h}_hours": float((notification <= h).mean())
                for h in [1, 3, 6]
            },
        },
        "overall": {
            "exact_facility_match_rate": float(review.facility_match.mean()),
            "exact_date_match_rate": float(review.date_exact.mean()),
            "exact_hour_match_rate": float(review.hour_exact.mean()),
            "exact_notification_timestamp_match_rate": float(
                review.notification_exact.mean()
            ),
            "exact_4_of_4_rate": float((review.exact_matches_out_of_4 == 4).mean()),
            "average_exact_matches": float(review.exact_matches_out_of_4.mean()),
            "tolerance_4_of_4_rate": float(
                (review.tolerance_matches_out_of_4 == 4).mean()
            ),
            "average_tolerance_matches": float(
                review.tolerance_matches_out_of_4.mean()
            ),
            "tolerance_facility_rate": float(review.facility_match.mean()),
            "tolerance_date_rate": float(review.date_tolerance.mean()),
            "tolerance_time_rate": float(review.time_tolerance.mean()),
            "tolerance_notification_rate": float(review.notification_tolerance.mean()),
        },
    }


def error_analysis(review: pd.DataFrame) -> dict:
    groups = {
        "activity": "activity_group",
        "season_transition": "season_transition",
        "preference_shift_proxy": "history_preference_shift",
        "actual_facility": "target_facility",
    }
    result = {}
    for name, column in groups.items():
        result[name] = {
            str(group): {
                "sample_count": len(rows),
                "facility_accuracy": float(rows.facility_match.mean()),
                "date_mae_days": float(rows.usage_interval_error_days.mean()),
                "time_mae_minutes": float(rows.usage_time_error_minutes.mean()),
                "notification_mae_hours": float(rows.notification_error_hours.mean()),
            }
            for group, rows in review.groupby(column, observed=True)
        }
    result["error_quantiles"] = {
        column: review[column].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).to_dict()
        for column in [
            "usage_interval_error_days",
            "usage_time_error_minutes",
            "notification_error_hours",
        ]
    }
    return result
