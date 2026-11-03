import pandas as pd
from openpyxl import load_workbook
from src.evaluate import review_predictions, metrics, error_analysis
from src.predict import predict_components
from src.reporting import write_workbook


def test_exact_and_tolerance_are_separate(examples):
    X, y, _, _ = examples
    y = y.iloc[:2].reset_index(drop=True)
    prediction = pd.DataFrame(
        {
            "predicted_facility": y.target_facility,
            "predicted_days_until_usage": y.days_until_usage,
            "predicted_model_lead_hours": y.lead_hours,
            "predicted_usage_timestamp": y.target_usage_timestamp,
            "predicted_notification_timestamp": y.target_booking_timestamp,
        }
    ).copy(deep=True)
    prediction.loc[1, "predicted_notification_timestamp"] += pd.Timedelta(seconds=1)
    review = review_predictions(y, prediction, X.iloc[:2])
    result = metrics(review)
    assert list(review.exact_matches_out_of_4) == [4, 3]
    assert list(review.tolerance_matches_out_of_4) == [4, 4]
    assert result["overall"]["exact_4_of_4_rate"] == 0.5
    assert result["overall"]["tolerance_4_of_4_rate"] == 1
    assert result["notification"]["mae_hours"] > 0


def test_excel_all_test_rows_metrics_and_styles(examples, trained, tmp_path):
    X, y, splits, _ = examples
    bundle, _ = trained
    indices = splits["test"]
    review = review_predictions(
        y.iloc[indices],
        predict_components(bundle, X.iloc[indices], y.prediction_cutoff.iloc[indices]),
        X.iloc[indices],
    )
    score = metrics(review)
    path = tmp_path / "review.xlsx"
    write_workbook(
        review,
        {kind: score for kind in ["ml", "resident", "global"]},
        error_analysis(review),
        {"bookings": 1400},
        path,
    )
    workbook = load_workbook(path)
    assert workbook.sheetnames == [
        "Prediction Review",
        "Performance Summary",
        "Detailed Metrics",
        "Error Analysis",
        "Dataset Insights",
    ]
    sheet = workbook["Prediction Review"]
    assert sheet.max_row == len(indices) + 1
    assert sheet.freeze_panes == "A2"
    assert sheet.auto_filter.ref
    assert len(sheet.conditional_formatting) >= 2
    headers = [cell.value for cell in sheet[1]]
    error_col = headers.index("Notification Time Error (Hours)") + 1
    assert (
        abs(sheet.cell(2, error_col).value - review.notification_error_hours.iloc[0])
        < 1e-10
    )
