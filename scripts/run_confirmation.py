import _bootstrap  # noqa: F401

"""Predeclared independent synthetic confirmation; never use its results to tune."""

from pathlib import Path
import shutil
import tempfile
from src.config import ROOT, GenerationConfig
from src.workflow import generate, train, evaluate
from src.utils import save_json
from src.artifacts import code_hashes

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="facility-confirmation-") as directory:
        root = Path(directory)
        generate(root, GenerationConfig(seed=20261103))
        train(
            root,
            evaluation_status="Predeclared independent synthetic seed 20261103; first post-freeze confirmation, no test-based selection",
        )
        report = evaluate(root)
        report["confirmation_seed"] = 20261103
        report["source_hashes"] = code_hashes()
        report["scope"] = (
            "Independent synthetic simulation, not prospective real-resident validation"
        )
        save_json(ROOT / "outputs/audit/confirmation_report.json", report)
        shutil.copyfile(
            root / "outputs/test_predictions.csv",
            ROOT / "outputs/audit/confirmation_predictions.csv",
        )
        shutil.copyfile(
            root / "outputs/facility_prediction_review.xlsx",
            ROOT / "outputs/audit/confirmation_review.xlsx",
        )
        print("Independent confirmation complete; seed=20261103")
