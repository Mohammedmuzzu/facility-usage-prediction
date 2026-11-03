import _bootstrap  # noqa: F401

"""Validate an append-only history update before atomically replacing the CSV."""

import argparse
from pathlib import Path
from src.config import ROOT
from src.artifacts import load_bundle, validate_history
from src.utils import load_bookings, file_hash, atomic_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Complete new history snapshot, including every original observation",
    )
    parser.add_argument(
        "--expected-current-sha256",
        required=True,
        help="Optimistic concurrency guard for current CSV",
    )
    args = parser.parse_args()
    destination = ROOT / "data/facility_bookings.csv"
    data = load_bookings(args.input)
    bundle = load_bundle(ROOT / "models/model_bundle.joblib")
    validate_history(data, bundle["metadata"]["history_contract"])
    from filelock import FileLock

    with FileLock(str(destination) + ".lock"):
        if file_hash(destination) != args.expected_current_sha256:
            raise ValueError("History changed since your read; reload before updating")
        from src.artifacts import history_contract

        validate_history(data, history_contract(load_bookings(destination)))
        atomic_text(destination, data.to_csv(index=False))
    print(
        "Append-only snapshot updated. Restart serving processes; evaluate is reserved for the original training snapshot."
    )
