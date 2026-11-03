import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault(
    "MPLCONFIGDIR",
    "/private/tmp/facility-test-matplotlib"
    if os.path.exists("/private/tmp")
    else "/tmp/facility-test-matplotlib",
)
import pytest
from src.config import GenerationConfig
from src.data_generator import generate_bookings
from src.dataset_builder import build_examples
from src.temporal_split import chronological_split
from src.train import fit_models


@pytest.fixture(scope="session")
def config():
    return GenerationConfig(residents=60, bookings=1400, months=12)


@pytest.fixture(scope="session")
def bookings(config):
    return generate_bookings(config)


@pytest.fixture(scope="session")
def examples(bookings):
    X, y = build_examples(bookings)
    splits, split = chronological_split(y)
    return X, y, splits, split


@pytest.fixture(scope="session")
def trained(examples, bookings, tmp_path_factory):
    X, y, splits, split = examples
    root = tmp_path_factory.mktemp("trained")
    from src.artifacts import history_contract

    metadata = {
        "dataset_sha256": "test-fixture",
        "split": split,
        "history_contract": history_contract(bookings),
    }
    bundle = fit_models(X, y, splits, metadata, root)
    return bundle, root
