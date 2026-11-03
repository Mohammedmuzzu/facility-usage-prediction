"""Optional validation-only permutation importance; does not tune or retrain."""
import _bootstrap
import joblib
from src.config import ROOT
from src.workflow import supervised
from src.explainability import facility_importance

if __name__ == "__main__":
    X, y, splits, _ = supervised()
    bundle = joblib.load(ROOT / "models/model_bundle.joblib")
    validation = splits["validation"]
    facility_importance(bundle, X.iloc[validation], y.iloc[validation]).to_csv(ROOT / "outputs/validation_feature_importance.csv", index=False)
