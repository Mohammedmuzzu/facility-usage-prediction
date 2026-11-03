"""Train-only feature references, batch drift diagnostics and release evidence."""

import numpy as np
import pandas as pd

MONITORED = [
    "total_bookings",
    "median_interval",
    "mean_lead_hours",
    *[f"facility_{i}_preference" for i in range(6)],
]


def reference_profile(X: pd.DataFrame) -> dict:
    result = {}
    for column in MONITORED:
        values = X[column].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8])).tolist()
        bins = [-np.inf, *edges, np.inf]
        counts = np.histogram(values, bins=bins)[0] + 0.5
        result[column] = {
            "interior_edges": edges,
            "proportions": (counts / counts.sum()).tolist(),
        }
    return result


def drift_report(reference: dict, X: pd.DataFrame) -> dict:
    if X.empty:
        return {"sample_count": 0, "features": {}, "alert": False}
    features = {}
    for column, profile in reference.items():
        counts = (
            np.histogram(X[column], bins=[-np.inf, *profile["interior_edges"], np.inf])[
                0
            ]
            + 0.5
        )
        actual = counts / counts.sum()
        expected = np.asarray(profile["proportions"])
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        features[column] = {"psi": psi, "investigate": psi > 0.2}
    return {
        "sample_count": len(X),
        "features": features,
        "alert": any(item["investigate"] for item in features.values()),
        "note": "PSI > 0.2 is an investigation heuristic, not proof of model failure; resident ageing itself changes cumulative features.",
    }


def coverage_report(
    bookings: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, splits: dict, split: dict
) -> tuple[pd.DataFrame, dict]:
    rows = y[
        [
            "booking_id",
            "resident_reference",
            "prediction_cutoff",
            "target_booking_timestamp",
            "simultaneous_target_count",
        ]
    ].copy()
    rows["partition"] = "purged_crossing_cutoff"
    for name, indices in splits.items():
        rows.loc[indices, "partition"] = name
    coverage = bookings[
        ["booking_id", "resident_reference", "booking_timestamp"]
    ].merge(
        rows.drop(columns="resident_reference"),
        on="booking_id",
        how="left",
        validate="one_to_one",
    )
    coverage["partition"] = coverage.partition.fillna(
        "initial_observation_or_simultaneous_non_target"
    )
    counts = coverage.partition.value_counts().to_dict()
    by_activity = []
    for group_name, mask in [
        ("sparse_0_4", X.total_bookings < 5),
        ("moderate_5_19", X.total_bookings.between(5, 19)),
        ("frequent_20_plus", X.total_bookings >= 20),
    ]:
        indices = set(np.flatnonzero(mask.to_numpy()).tolist())
        retained = {
            name: len(indices.intersection(values.tolist()))
            for name, values in splits.items()
        }
        by_activity.append(
            {
                "group": group_name,
                "sequential_examples": len(indices),
                "retained": retained,
                "purged": len(indices) - sum(retained.values()),
            }
        )
    summary = {
        "all_booking_records": len(bookings),
        "sequential_examples": len(y),
        "record_partition_counts": counts,
        "censored_final_creation_cutoffs": int(bookings.resident_reference.nunique()),
        "simultaneous_next_booking_examples": int(
            (y.simultaneous_target_count > 1).sum()
        ),
        "tie_policy": "At a tied next creation event choose lexicographically smallest booking_id; all ties at a cutoff are visible",
        "activity_coverage": by_activity,
        "partition_boundaries": split,
    }
    return coverage, summary


def block_bootstrap(
    review: pd.DataFrame, repetitions: int = 200, seed: int = 42
) -> dict:
    """Percentile intervals resampling residents, preserving within-resident dependence."""
    grouped = [group for _, group in review.groupby("resident_reference")]
    if len(grouped) < 2:
        return {
            "resident_count": len(grouped),
            "intervals": {},
            "note": "Too few resident clusters",
        }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = [grouped[i] for i in rng.integers(0, len(grouped), len(grouped))]
        size = sum(len(group) for group in sampled)
        values.append(
            [
                sum(group.facility_match.sum() for group in sampled) / size,
                sum(group.days_regression_error.sum() for group in sampled) / size,
                sum(group.usage_time_error_minutes.sum() for group in sampled) / size,
                sum(group.notification_error_hours.sum() for group in sampled) / size,
            ]
        )
    intervals = np.quantile(values, [0.025, 0.975], axis=0)
    return {
        "resident_count": len(grouped),
        "repetitions": repetitions,
        "seed": seed,
        "intervals": {
            name: {
                "lower_95": float(intervals[0, i]),
                "upper_95": float(intervals[1, i]),
            }
            for i, name in enumerate(
                [
                    "facility_accuracy",
                    "days_mae",
                    "circular_minutes_mae",
                    "notification_hours_mae",
                ]
            )
        },
        "note": "Resident-cluster percentile bootstrap; descriptive uncertainty on synthetic observations, not real-world or causal validity.",
    }


def release_gates(metrics: dict) -> dict:
    # No organization-specific business threshold is invented. Explicitly require
    # prospective real-data/notification evidence before an effectiveness release.
    return {
        "technical_artifacts": "validated separately by manifest, tests and CI",
        "production_business_approved": False,
        "blocking_external_requirements": [
            "Real booking/availability integration",
            "Organization-approved SLOs and quality thresholds",
            "Prospective chronological real-data validation",
            "Notification engagement/conversion experiment",
        ],
        "observed_tolerance_4_of_4_rate": metrics["champion"]["overall"][
            "tolerance_4_of_4_rate"
        ],
        "decision": "Engineering candidate only; do not claim optimal reminders or unrestricted production effectiveness",
    }
