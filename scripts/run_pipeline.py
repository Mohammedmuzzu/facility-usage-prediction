import _bootstrap
from src.workflow import run

if __name__ == "__main__":
    report = run()
    print("Pipeline complete. Test records:", report["metrics"]["ml"]["sample_count"])
    print("Review workbook: outputs/facility_prediction_review.xlsx")
