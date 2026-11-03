import _bootstrap  # noqa: F401

"""Activate a verified immutable release, then restart serving processes."""

import argparse
from src.config import ROOT
from src.artifacts import activate_release

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("release_id")
    args = parser.parse_args()
    activate_release(ROOT / "models", args.release_id)
    print("Release activated. Restart API/dashboard processes to load it.")
