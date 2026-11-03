"""Small predeclared CPU model grid, selected only on chronological validation."""

import importlib.metadata
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.utils.class_weight import compute_sample_weight
from src.baselines import Baselines
from src.config import SEED
from src.utils import minutes_of_day, circular_error
from src.temporal_split import validate_splits
from src.artifacts import publish_bundle
from src.config import FACILITIES
from src.monitoring import reference_profile


def pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    categories = X.select_dtypes(include=["object", "str", "string"]).columns.tolist()
    transform = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categories,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", transform), ("estimator", estimator)])


def fit_models(
    X: pd.DataFrame, y: pd.DataFrame, splits: dict, metadata: dict, root: Path
) -> dict:
    validate_splits(y, splits)
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("Feature and target rows must align")
    train, validation = splits["train"], splits["validation"]
    X_train, X_validation = X.iloc[train], X.iloc[validation]
    y_train, y_validation = y.iloc[train], y.iloc[validation]
    if y_train.target_facility.nunique() < 2:
        raise ValueError("Facility training requires at least two classes")
    baseline = Baselines().fit(X_train, y_train)
    angles_train = minutes_of_day(y_train.target_usage_timestamp) * 2 * np.pi / 1440
    angles_validation = (
        minutes_of_day(y_validation.target_usage_timestamp) * 2 * np.pi / 1440
    )
    targets = {
        "facility": (y_train.target_facility, y_validation.target_facility),
        "usage_days": (y_train.days_until_usage, y_validation.days_until_usage),
        "usage_sin": (np.sin(angles_train), np.sin(angles_validation)),
        "usage_cos": (np.cos(angles_train), np.cos(angles_validation)),
        "lead_hours": (y_train.lead_hours, y_validation.lead_hours),
    }
    candidates = [
        {"max_leaf_nodes": 15, "max_iter": 140, "l2_regularization": 5.0},
        {"max_leaf_nodes": 31, "max_iter": 200, "l2_regularization": 10.0},
    ]
    selected, audit, configurations, time_candidates = {}, {}, {}, {}
    # No early stopping: sklearn's random internal validation is deliberately disabled.
    for name, (train_target, validation_target) in targets.items():
        best_score, best_model, records = float("inf"), None, []
        for index, config in enumerate(candidates):
            common = dict(
                **config, learning_rate=0.06, random_state=SEED, early_stopping=False
            )
            if name == "facility":
                estimator = HistGradientBoostingClassifier(**common)
                weight = np.sqrt(compute_sample_weight("balanced", train_target))
                model = pipeline(X_train, estimator).fit(
                    X_train, train_target, estimator__sample_weight=weight
                )
            else:
                estimator = HistGradientBoostingRegressor(
                    **common,
                    loss="squared_error"
                    if name in ["usage_sin", "usage_cos"]
                    else "absolute_error",
                )
                model = pipeline(X_train, estimator).fit(X_train, train_target)
            prediction = model.predict(X_validation)
            score = (
                -f1_score(
                    validation_target,
                    prediction,
                    labels=list(FACILITIES),
                    average="macro",
                    zero_division=0,
                )
                if name == "facility"
                else mean_absolute_error(validation_target, prediction)
            )
            records.append(
                {
                    "candidate": index,
                    "configuration": config,
                    "validation_loss": float(score),
                }
            )
            if name in ["usage_sin", "usage_cos"]:
                time_candidates.setdefault(name, []).append(model)
            if score < best_score:
                best_score, best_model = score, model
                configurations[name] = config
        selected[name], audit[name] = best_model, records
    # Select the sine/cosine pair jointly by circular error on validation only.
    paired_scores = []
    for i, sine_model in enumerate(time_candidates["usage_sin"]):
        for j, cosine_model in enumerate(time_candidates["usage_cos"]):
            minute_prediction = (
                np.mod(
                    np.arctan2(
                        sine_model.predict(X_validation),
                        cosine_model.predict(X_validation),
                    ),
                    2 * np.pi,
                )
                * 1440
                / (2 * np.pi)
            )
            paired_scores.append(
                (
                    float(
                        circular_error(
                            minutes_of_day(y_validation.target_usage_timestamp),
                            minute_prediction,
                        ).mean()
                    ),
                    i,
                    j,
                )
            )
    _, sine_index, cosine_index = min(paired_scores)
    for name, index in [("usage_sin", sine_index), ("usage_cos", cosine_index)]:
        selected[name] = time_candidates[name][index]
        configurations[name] = candidates[index]
    audit["time_pairs"] = [
        {"circular_mae_minutes": score, "sine_candidate": i, "cosine_candidate": j}
        for score, i, j in paired_scores
    ]
    validation_minutes = (
        np.mod(
            np.arctan2(
                selected["usage_sin"].predict(X_validation),
                selected["usage_cos"].predict(X_validation),
            ),
            2 * np.pi,
        )
        * 1440
        / (2 * np.pi)
    )
    metadata.update(
        {
            "seed": SEED,
            "ready_at": metadata["split"]["test_start"],
            "fit_indices": train.tolist(),
            "selection_indices": validation.tolist(),
            "test_used_for_fit_or_selection": False,
            "feature_names": X.columns.tolist(),
            "categorical_features": X.select_dtypes(
                include=["object", "str", "string"]
            ).columns.tolist(),
            "selected_configurations": configurations,
            "validation_candidates": audit,
            "estimator_parameters": {
                name: model.named_steps["estimator"].get_params()
                for name, model in selected.items()
            },
            "validation_hour_circular_mae_minutes": float(
                circular_error(
                    minutes_of_day(y_validation.target_usage_timestamp),
                    validation_minutes,
                ).mean()
            ),
            "python": platform.python_version(),
            "dependencies": {
                p: importlib.metadata.version(p)
                for p in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "matplotlib",
                    "joblib",
                    "openpyxl",
                    "fastapi",
                    "streamlit",
                ]
            },
            "model_family": "HistGradientBoosting; weighted multiclass classification, absolute-error day/lead regression, sine/cosine time regression",
            "label_availability": "Scheduled usage and lead are known when booking is created; attendance is not modelled",
        }
    )
    bundle = {"models": selected, "baseline": baseline, "metadata": metadata}
    metadata["monitoring_reference"] = reference_profile(X_train)
    select_routes(bundle, X_validation, y_validation)
    publish_bundle(bundle, root)
    return bundle


def select_routes(bundle: dict, X: pd.DataFrame, y: pd.DataFrame) -> None:
    """Compare heuristics with ML only on validation; never infer a test winner."""
    from src.predict import raw_components, assemble_predictions

    candidates = {
        kind: raw_components(bundle, X, kind) for kind in ["resident", "ml", "global"]
    }
    routes, scores = {}, {}
    for component in ["facility", "days", "minutes"]:
        losses = {}
        for kind, values in candidates.items():
            if component == "facility":
                loss = -f1_score(
                    y.target_facility,
                    values[component],
                    labels=list(FACILITIES),
                    average="macro",
                    zero_division=0,
                )
            elif component == "days":
                loss = mean_absolute_error(
                    y.days_until_usage, np.clip(values[component], 0, 365)
                )
            else:
                loss = circular_error(
                    minutes_of_day(y.target_usage_timestamp), values[component]
                ).mean()
            losses[kind] = float(loss)
        routes[component] = min(losses, key=losses.get)
        scores[component] = losses
    selected = {
        component: candidates[route][component] for component, route in routes.items()
    }
    notification_losses = {}
    for kind, values in candidates.items():
        prediction = assemble_predictions(
            y.prediction_cutoff, {**selected, "lead_hours": values["lead_hours"]}
        )
        error = (
            prediction.predicted_notification_timestamp
            - y.target_booking_timestamp.reset_index(drop=True)
        ).dt.total_seconds().abs() / 3600
        notification_losses[kind] = float(error.mean())
    routes["lead_hours"] = min(notification_losses, key=notification_losses.get)
    scores["lead_hours"] = notification_losses
    bundle["metadata"].update(
        {
            "routes": routes,
            "route_validation_losses": scores,
            "route_objectives": {
                "facility": "negative macro F1",
                "days": "elapsed-day MAE",
                "minutes": "circular minute MAE",
                "lead_hours": "end-to-end booking-time proxy MAE",
            },
        }
    )
    from src.evaluate import review_predictions

    prediction = assemble_predictions(
        y.prediction_cutoff,
        {
            component: candidates[route][component]
            for component, route in routes.items()
        },
    )
    review = review_predictions(y, prediction, X)
    bundle["metadata"]["validation_error_guides"] = {
        name: float(review[column].quantile(0.9))
        for name, column in [
            ("elapsed_days_p90_absolute_error", "days_regression_error"),
            ("usage_minutes_p90_circular_error", "usage_time_error_minutes"),
            ("notification_hours_p90_absolute_error", "notification_error_hours"),
        ]
    }
