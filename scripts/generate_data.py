import _bootstrap  # noqa: F401
import argparse
from src.config import GenerationConfig
from src.workflow import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a domain-aware synthetic booking dataset"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--residents", type=int, default=500)
    parser.add_argument("--bookings", type=int, default=12500)
    parser.add_argument("--months", type=int, default=18)
    args = parser.parse_args()
    generate(
        config=GenerationConfig(
            seed=args.seed,
            residents=args.residents,
            bookings=args.bookings,
            months=args.months,
        )
    )
