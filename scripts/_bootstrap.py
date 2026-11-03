"""Allow documented direct script execution without installing the package."""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
)
os.environ.setdefault("OMP_NUM_THREADS", "2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
