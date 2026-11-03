"""Central configuration. Calendar arithmetic is consistently Asia/Kolkata."""

from dataclasses import asdict, dataclass
from pathlib import Path
import os

ROOT = Path(
    os.environ.get("FACILITY_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)
FACILITIES = (
    "Gym",
    "Swimming Pool",
    "Badminton Court",
    "Tennis Court",
    "Clubhouse",
    "Multipurpose Hall",
)
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

    def __post_init__(self) -> None:
        import math
        import pandas as pd

        if any(
            type(x) is not int
            for x in [self.residents, self.bookings, self.months, self.seed]
        ):
            raise ValueError("Population, bookings, months and seed must be integers")
        if (
            not 2 <= self.residents <= 9999
            or not self.residents <= self.bookings <= 2_000_000
        ):
            raise ValueError(
                "Require 2–9999 residents and resident_count–2,000,000 bookings"
            )
        if not 3 <= self.months <= 120 or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Require 3–120 months and an unsigned 32-bit seed")
        if any(
            not math.isfinite(x) or not 0 <= x <= 1
            for x in [self.drift_fraction, self.exploration, self.sparse_fraction]
        ):
            raise ValueError("Behaviour fractions must be finite and within [0, 1]")
        if len(self.popularity) != 6 or any(
            not math.isfinite(x) or x <= 0 for x in self.popularity
        ):
            raise ValueError("Require six finite positive popularity weights")
        date = pd.Timestamp(self.start)
        if (
            date.tzinfo is not None
            or date != date.normalize()
            or date.day != 1
            or not 2000 <= date.year <= 2090
        ):
            raise ValueError(
                "Start must be the first day of a month, midnight, year 2000–2090, without an offset"
            )

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_directories(root: Path = ROOT) -> None:
    for directory in ("data", "models", "outputs/figures", "docs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
