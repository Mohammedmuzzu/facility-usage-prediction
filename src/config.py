"""Central configuration. Calendar arithmetic is consistently Asia/Kolkata."""
from dataclasses import asdict, dataclass
from pathlib import Path
import os

ROOT = Path(os.environ.get("FACILITY_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
FACILITIES = ("Gym", "Swimming Pool", "Badminton Court", "Tennis Court", "Clubhouse", "Multipurpose Hall")
TIMEZONE = "Asia/Kolkata"
SEED = 42


@dataclass(frozen=True)
class GenerationConfig:
    residents: int = 500
    bookings: int = 12500
    months: int = 18
    start: str = "2025-01-01"
    seed: int = SEED
    drift_fraction: float = 0.28
    exploration: float = 0.08
    sparse_fraction: float = 0.12
    popularity: tuple[float, ...] = (1.8, 1.2, 1.1, 0.8, 0.65, 0.4)

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_directories(root: Path = ROOT) -> None:
    for directory in ("data", "models", "outputs/figures", "docs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
