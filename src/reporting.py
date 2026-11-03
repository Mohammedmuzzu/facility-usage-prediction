"""Programmatic Excel review, diagnostic figures, and evidence-based Markdown report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils import get_column_letter
from src.config import FACILITIES

NAVY = "16324F"
GREEN, RED, AMBER = "D9EAD3", "F4CCCC", "FFF2CC"


def flatten(value: dict, prefix: str = "") -> list[tuple]:
    rows = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            rows.extend(flatten(item, name))
        else:
            rows.append((name, json.dumps(item) if isinstance(item, list) else item))
    return rows


def explain_error(row: pd.Series) -> str:
    messages = []
    if not row.facility_match:
        messages.append("Facility preference mismatch")
    if row.history_count < 5:
        messages.append("Sparse resident history")
    if row.usage_date_error_days > 1:
        messages.append("Next-booking interval outside one calendar day")
    if row.usage_time_error_minutes > 60:
        messages.append("Usage time outside 60-minute tolerance")
    if row.notification_error_hours > 3:
        messages.append("Booking-time proxy outside 3-hour tolerance")
    if row.season_transition:
        messages.append("Usage crosses a calendar-quarter boundary")
    if row.history_preference_shift:
        messages.append("Historical recent and lifetime preferences differ")
    return "; ".join(messages) or "Within documented tolerances"


def write_workbook(review: pd.DataFrame, metrics: dict, analysis: dict, metadata: dict, path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Prediction Review"
    columns = {
        "booking_id": "Record Reference", "resident_reference": "Resident Reference", "prediction_cutoff": "Prediction Cutoff",
        "predicted_facility": "Predicted Facility", "target_facility": "Actual Facility", "facility_match": "Facility Match",
        "predicted_usage_timestamp": "Predicted Usage Timestamp", "target_usage_timestamp": "Actual Usage Timestamp",
        "usage_date_error_days": "Usage Date Error (Days)", "date_exact": "Exact Date Match", "date_tolerance": "Date Within 1 Day",
        "usage_time_error_minutes": "Usage Time Error (Minutes)", "hour_exact": "Exact Hour Match", "time_tolerance": "Time Within 60 Minutes",
        "predicted_notification_timestamp": "Predicted Notification Timestamp", "target_booking_timestamp": "Actual Booking Timestamp",
        "notification_error_hours": "Notification Time Error (Hours)", "notification_exact": "Exact Notification Match",
        "notification_tolerance": "Notification Within 3 Hours", "exact_matches_out_of_4": "Exact Matches Out of 4",
        "tolerance_matches_out_of_4": "Tolerance Matches Out of 4", "overall_result": "Overall Result", "history_count": "Historical Bookings",
    }
    sheet.append(list(columns.values()))
    date_keys = {"prediction_cutoff", "predicted_usage_timestamp", "target_usage_timestamp", "predicted_notification_timestamp", "target_booking_timestamp"}
    match_keys = {"facility_match": ("facility_match", None), "date_exact": ("date_exact", "date_tolerance"),
                  "hour_exact": ("hour_exact", "time_tolerance"), "notification_exact": ("notification_exact", "notification_tolerance")}
    for _, row in review.iterrows():
        values = []
        for key in columns:
            value = row[key]
            if key in date_keys:
                value = value.tz_localize(None).to_pydatetime(warn=False)
            elif isinstance(value, np.generic):
                value = value.item()
            values.append(value)
        sheet.append(values)
        for index, key in enumerate(columns, 1):
            cell = sheet.cell(sheet.max_row, index)
            if key in date_keys:
                cell.number_format = "yyyy-mm-dd hh:mm:ss.000"
            elif key in match_keys:
                exact_key, tolerance_key = match_keys[key]
                color = GREEN if row[exact_key] else AMBER if tolerance_key and row[tolerance_key] else RED
                cell.fill = PatternFill("solid", fgColor=color)
            elif key == "overall_result":
                cell.fill = PatternFill("solid", fgColor={"Correct": GREEN, "Within tolerance": AMBER, "Incorrect": RED}[row[key]])
    # Separate date/time columns make direct review possible without interpreting a combined timestamp.
    for heading, key, fmt in [("Predicted Usage Date", "predicted_usage_timestamp", "yyyy-mm-dd"),
                              ("Actual Usage Date", "target_usage_timestamp", "yyyy-mm-dd"),
                              ("Predicted Usage Time", "predicted_usage_timestamp", "hh:mm:ss.000"),
                              ("Actual Usage Time", "target_usage_timestamp", "hh:mm:ss.000")]:
        column = sheet.max_column + 1
        sheet.cell(1, column, heading)
        for row_index, value in enumerate(review[key], 2):
            cell = sheet.cell(row_index, column, value.tz_localize(None).to_pydatetime(warn=False))
            cell.number_format = fmt
    for key in ["exact_matches_out_of_4", "tolerance_matches_out_of_4"]:
        col = get_column_letter(list(columns).index(key) + 1)
        area = f"{col}2:{col}{sheet.max_row}"
        sheet.conditional_formatting.add(area, CellIsRule(operator="equal", formula=["4"], fill=PatternFill("solid", fgColor=GREEN)))
        sheet.conditional_formatting.add(area, CellIsRule(operator="lessThan", formula=["4"], fill=PatternFill("solid", fgColor=RED)))
    summary = workbook.create_sheet("Performance Summary")
    summary.append(["Metric", "ML", "Resident Baseline", "Global Baseline"])
    summary.append(["All dates and times use Asia/Kolkata. Excel timestamps omit the timezone suffix."])
    paths = [("sample_count",), ("overall", "exact_4_of_4_rate"), ("overall", "tolerance_4_of_4_rate"),
             ("overall", "average_exact_matches"), ("overall", "average_tolerance_matches"),
             ("facility", "accuracy"), ("facility", "macro_f1"), ("usage_day", "mae_days"),
             ("usage_hour", "circular_mae_minutes"), ("notification", "mae_hours")]
    for keys in paths:
        values = []
        for kind in ["ml", "resident", "global"]:
            value = metrics[kind]
            for key in keys:
                value = value[key]
            values.append(value)
        summary.append([" / ".join(keys), *values])
    detailed = workbook.create_sheet("Detailed Metrics")
    detailed.append(["Model", "Metric", "Value"])
    for kind, values in metrics.items():
        for key, value in flatten(values):
            detailed.append([kind, key, value])
    for key, value in flatten(analysis):
        detailed.append(["error_analysis", key, value])
    errors = workbook.create_sheet("Error Analysis")
    errors.append(["Record", "Resident", "Explanation", "Facility Match", "Date Error (Days)", "Time Error (Minutes)", "Notification Error (Hours)"])
    worst = review[review.overall_result == "Incorrect"].sort_values(["tolerance_matches_out_of_4", "notification_error_hours"], ascending=[True, False]).head(100)
    for _, row in worst.iterrows():
        errors.append([row.booking_id, row.resident_reference, explain_error(row), bool(row.facility_match),
                       int(row.usage_date_error_days), float(row.usage_time_error_minutes), float(row.notification_error_hours)])
    insights = workbook.create_sheet("Dataset Insights")
    insights.append(["Dataset Metric", "Value"])
    for key, value in flatten(metadata):
        insights.append([key, value])
    for current in workbook:
        current.freeze_panes = "A2"
        current.auto_filter.ref = current.dimensions
        current.sheet_view.showGridLines = False
        current.row_dimensions[1].height = 34
        for cell in current[1]:
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = PatternFill("solid", fgColor=NAVY)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for column in current.columns:
            width = min(55, max(18, max(len(str(cell.value or "")) for cell in list(column)[:100]) + 2))
            current.column_dimensions[column[0].column_letter].width = width
    errors.column_dimensions["C"].width = 100
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def create_figures(data: pd.DataFrame, review: pd.DataFrame, metrics: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(destination / f"{name}.png", bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(9, 4))
    data.facility_reference.value_counts().reindex(FACILITIES).plot.bar(color="#327A9F")
    plt.ylabel("Bookings"); plt.title("Facility popularity"); save("facility_distribution")
    plt.figure(figsize=(7, 4))
    plt.hist(data.resident_reference.value_counts(), bins=30, color="#327A9F")
    plt.xlabel("Bookings per resident"); plt.ylabel("Residents"); save("resident_activity")
    plt.figure(figsize=(7, 4))
    plt.hist((data.usage_timestamp - data.booking_timestamp).dt.total_seconds() / 3600, bins=50, color="#327A9F")
    plt.xlabel("Advance booking lead (hours)"); save("lead_time_distribution")
    plt.figure(figsize=(8, 4))
    pd.crosstab(data.usage_timestamp.dt.hour, data.facility_reference).plot(ax=plt.gca())
    plt.xlabel("Scheduled usage hour (Asia/Kolkata)"); plt.ylabel("Bookings"); save("usage_hour_distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = np.asarray(metrics["ml"]["facility"]["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues")
    labels = [x.replace(" ", "\n") for x in FACILITIES]
    ax.set_xticks(range(6), labels, fontsize=8); ax.set_yticks(range(6), labels, fontsize=8)
    ax.set_xlabel("Predicted facility"); ax.set_ylabel("Actual facility")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    save("facility_confusion_matrix")
    plt.figure(figsize=(6, 5))
    plt.scatter(review.days_until_usage, review.predicted_days_until_usage, alpha=.25, s=9)
    upper = max(review.days_until_usage.max(), review.predicted_days_until_usage.max())
    plt.plot([0, upper], [0, upper], "k--")
    plt.xlabel("Actual days until scheduled usage"); plt.ylabel("Predicted days"); save("usage_interval_scatter")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, column, label in zip(axes, ["days_regression_error", "usage_time_error_minutes", "notification_error_hours"], ["Day error (days)", "Circular time error (minutes)", "Notification proxy error (hours)"]):
        ax.hist(review[column], bins=35, color="#327A9F"); ax.set_xlabel(label)
    save("prediction_error_distributions")
    plt.figure(figsize=(7, 4))
    review.groupby("activity_group", observed=True).facility_match.mean().plot.bar(color="#327A9F")
    plt.ylim(0, 1); plt.ylabel("Facility accuracy"); save("performance_by_activity")


def write_evaluation_markdown(metrics: dict, analysis: dict, split: dict, destination: Path) -> None:
    ml = metrics["ml"]
    lines = ["# Evaluation report", "", "Generated from saved models and actual chronological holdout predictions; no metrics are hand-entered.", "",
             f"Test records: **{ml['sample_count']}**. Purged crossing-cutoff examples: **{split['purged_examples']}**.", "",
             "| Output / metric | ML | Resident baseline | Global baseline | ML improvement vs resident |",
             "|---|---:|---:|---:|---:|"]
    for label, group, key, higher in [("Facility accuracy", "facility", "accuracy", True), ("Facility macro F1", "facility", "macro_f1", True),
                                     ("Days MAE", "usage_day", "mae_days", False), ("Circular time MAE (minutes)", "usage_hour", "circular_mae_minutes", False),
                                     ("Notification proxy MAE (hours)", "notification", "mae_hours", False)]:
        values = [metrics[k][group][key] for k in ["ml", "resident", "global"]]
        difference = values[0] - values[1] if higher else values[1] - values[0]
        lines.append(f"| {label} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {difference:+.4f} |")
    overall = ml["overall"]
    lines += ["", f"Exact 4/4 match: **{overall['exact_4_of_4_rate']:.2%}**. Tolerance 4/4 match: **{overall['tolerance_4_of_4_rate']:.2%}**.",
              f"Average exact matches: **{overall['average_exact_matches']:.3f}/4**. Average tolerance matches: **{overall['average_tolerance_matches']:.3f}/4**.", "",
              "Exact means facility, calendar date, clock-hour bucket, and full notification timestamp equality. Tolerance means facility exact, date within one calendar day, time within 60 circular minutes, notification within 3 elapsed hours. No underlying predictions are rounded for equality.", "",
              "Day-regression metrics use raw predicted elapsed days; calendar-date metrics use the assembled usage date. Notification errors include errors propagated from predicted usage and the constrained lead model. They measure a historical booking-time proxy only.", "", "## Chronological boundaries", "", "```json", json.dumps(split, indent=2), "```", "", "## Error analysis", "",
              "The following groups are diagnostic slices, not causal explanations. Preference shift is a history-only proxy (L1 distance > 0.12 between recent and lifetime smoothed facility shares), not access to synthetic persona labels.", "",
              "| Activity group | Samples | Facility accuracy | Date MAE (days) | Time MAE (minutes) | Notification MAE (hours) |", "|---|---:|---:|---:|---:|---:|"]
    for name, values in analysis["activity"].items():
        lines.append(f"| {name} | {values['sample_count']} | {values['facility_accuracy']:.2%} | {values['date_mae_days']:.2f} | {values['time_mae_minutes']:.2f} | {values['notification_mae_hours']:.2f} |")
    worst = min(analysis["actual_facility"].items(), key=lambda item: item[1]["facility_accuracy"])
    lines += ["", f"Lowest facility recall occurs for **{worst[0]}** ({worst[1]['facility_accuracy']:.2%}, n={worst[1]['sample_count']}). Class imbalance, exploration, multimodal usage hours, and changing preferences make rare-facility and timing predictions difficult.",
              "Notification errors can remain large even when facility prediction is correct: an error of one day in scheduled usage can move the implied booking-time proxy by roughly 24 hours. Exact notification equality is intentionally strict and is generally unlikely for a continuous regression output.",
              "", "Full metrics, all three model variants, subgroup counts, seasonal-transition and drift-proxy slices, and error quantiles are in `outputs/evaluation_report.json`. Every final test row is in the workbook and CSV.", "",
              "![Facility confusion matrix](../outputs/figures/facility_confusion_matrix.png)", "![Usage intervals](../outputs/figures/usage_interval_scatter.png)", "![Errors](../outputs/figures/prediction_error_distributions.png)"]
    destination.write_text("\n".join(lines) + "\n")
