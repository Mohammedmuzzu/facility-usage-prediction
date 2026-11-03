"""Composable workflow stages; evaluation/prediction never call fit_models."""
import json
import logging
from pathlib import Path
import joblib
from src.config import ROOT, GenerationConfig, ensure_directories
from src.data_generator import generate_bookings
from src.data_validation import validate_bookings
from src.dataset_builder import build_examples
from src.temporal_split import chronological_split
from src.train import fit_models
from src.predict import predict_components
from src.evaluate import metrics, review_predictions, error_analysis
from src.reporting import write_workbook, create_figures, write_evaluation_markdown
from src.utils import load_bookings, save_json, file_hash

LOG = logging.getLogger(__name__)


def generate(root: Path = ROOT, config: GenerationConfig = GenerationConfig()) -> dict:
    ensure_directories(root)
    data = generate_bookings(config)
    metadata = validate_bookings(data, config)
    path = root / "data/facility_bookings.csv"
    data.to_csv(path, index=False)
    metadata["dataset_sha256"] = file_hash(path)
    save_json(root / "data/dataset_metadata.json", metadata)
    save_json(root / "outputs/dataset_profile.json", metadata)
    (root / "docs/dataset_profile.md").write_text("# Dataset profile\n\n```json\n" + json.dumps(metadata, indent=2) + "\n```\n")
    LOG.info("Generated and validated %d bookings for %d residents", len(data), metadata["residents"])
    return metadata


def supervised(root: Path = ROOT):
    path = root / "data/facility_bookings.csv"
    digest = file_hash(path)
    feature_version = "|".join(file_hash(Path(__file__).parent / name) for name in ["feature_engineering.py", "dataset_builder.py", "temporal_split.py"])
    cache = root / ".cache/supervised.joblib"
    if cache.exists():
        saved = joblib.load(cache)
        if saved[0] == (digest, feature_version):
            return saved[1:]
    data = load_bookings(path)
    validate_bookings(data)
    LOG.info("Building historical feature vectors (same builder as inference)")
    X, y = build_examples(data)
    splits, split = chronological_split(y)
    cache.parent.mkdir(exist_ok=True)
    joblib.dump(((digest, feature_version), X, y, splits, split), cache, compress=3)
    return X, y, splits, split


def train(root: Path = ROOT) -> dict:
    X, y, splits, split = supervised(root)
    metadata = {"dataset_sha256": file_hash(root / "data/facility_bookings.csv"), "split": split}
    LOG.info("Fitting candidates on train=%d; selecting on validation=%d", len(splits["train"]), len(splits["validation"]))
    bundle = fit_models(X, y, splits, metadata, root / "models")
    save_json(root / "outputs/split_boundaries.json", split)
    return bundle["metadata"]


def evaluate(root: Path = ROOT) -> dict:
    X, y, splits, split = supervised(root)
    bundle = joblib.load(root / "models/model_bundle.joblib")
    if bundle["metadata"]["dataset_sha256"] != file_hash(root / "data/facility_bookings.csv"):
        raise ValueError("Model/data hash mismatch; explicitly retrain before evaluating")
    test = splits["test"]
    X_test, y_test = X.iloc[test].reset_index(drop=True), y.iloc[test].reset_index(drop=True)
    reviews, all_metrics = {}, {}
    for kind in ["ml", "resident", "global"]:
        predicted = predict_components(bundle, X_test, y_test.prediction_cutoff, kind)
        reviews[kind] = review_predictions(y_test, predicted, X_test)
        all_metrics[kind] = metrics(reviews[kind])
        reviews[kind].to_csv(root / f"outputs/{'test_predictions' if kind == 'ml' else kind + '_baseline_predictions'}.csv", index=False)
    analysis = error_analysis(reviews["ml"])
    report = {"metrics": all_metrics, "error_analysis": analysis, "split": split,
              "dataset_sha256": bundle["metadata"]["dataset_sha256"],
              "tolerances": {"facility": "exact", "usage_date": "1 calendar day", "usage_time": "60 circular minutes", "notification": "3 elapsed hours"}}
    save_json(root / "outputs/evaluation_metrics.json", all_metrics)
    save_json(root / "outputs/evaluation_report.json", report)
    dataset_metadata = json.loads((root / "data/dataset_metadata.json").read_text())
    write_workbook(reviews["ml"], all_metrics, analysis, dataset_metadata, root / "outputs/facility_prediction_review.xlsx")
    create_figures(load_bookings(root / "data/facility_bookings.csv"), reviews["ml"], all_metrics, root / "outputs/figures")
    write_evaluation_markdown(all_metrics, analysis, split, root / "docs/evaluation_report.md")
    LOG.info("Evaluated %d unseen chronological test examples; workbook and reports saved", len(test))
    return report


def run(root: Path = ROOT) -> dict:
    generate(root)
    train(root)
    return evaluate(root)
