"""Generate the full test review without retraining; optionally predict a new resident."""
import _bootstrap
import argparse
import json
from src.config import ROOT
from src.predict import PredictionEngine
from src.utils import load_bookings
from src.workflow import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resident")
    parser.add_argument("--cutoff", help="Timezone-aware ISO timestamp")
    args = parser.parse_args()
    if args.resident:
        if not args.cutoff:
            parser.error("--resident requires --cutoff")
        engine = PredictionEngine(ROOT / "models/model_bundle.joblib", load_bookings(ROOT / "data/facility_bookings.csv"))
        print(json.dumps(engine.predict(args.resident, args.cutoff), indent=2))
    else:
        evaluate()
