import _bootstrap  # noqa: F401
from src.config import ROOT
from src.workflow import supervised
from src.artifacts import load_bundle
from src.robustness import temporal_backtests, cold_start_review
from src.utils import load_bookings, save_json

if __name__ == "__main__":
    X, y, _, split = supervised()
    bundle = load_bundle(ROOT / "models/model_bundle.joblib")
    report = temporal_backtests(X, y, split["test_start"])
    report["cold_start_cohort"] = cold_start_review(
        load_bookings(ROOT / "data/facility_bookings.csv"), bundle, split["test_start"]
    )
    save_json(ROOT / "outputs/robustness_report.json", report)
    print("Development-only windows:", len(report["windows"]))
