"""Domain-aware synthetic scheduled bookings. Latent personas never leave the generator."""

import numpy as np
import pandas as pd
from src.config import FACILITIES, TIMEZONE, GenerationConfig


def generate_bookings(config: GenerationConfig = GenerationConfig()) -> pd.DataFrame:
    if config.residents < 2 or config.bookings < config.residents or config.months < 3:
        raise ValueError("Need >=2 residents, >=1 booking per resident, and >=3 months")
    if len(config.popularity) != len(FACILITIES) or min(config.popularity) <= 0:
        raise ValueError("Facility popularity must contain six positive weights")
    if not all(
        0 <= x <= 1
        for x in (config.exploration, config.drift_fraction, config.sparse_fraction)
    ):
        raise ValueError("Behaviour fractions must be in [0, 1]")
    rng = np.random.default_rng(config.seed)
    start = pd.Timestamp(config.start, tz=TIMEZONE)
    end = start + pd.DateOffset(months=config.months)
    total_days = (end - start).days
    sparse = rng.random(config.residents) < config.sparse_fraction
    rates = rng.gamma(1.6, 1, config.residents)
    rates[sparse] *= 0.035
    counts = np.ones(config.residents, dtype=int) + rng.multinomial(
        config.bookings - config.residents, rates / rates.sum()
    )
    personas = np.array(
        [
            [11, 1, 0.7, 0.5, 0.3, 0.2],
            [1, 10, 0.5, 0.5, 1, 0.3],
            [1, 1, 8, 6, 0.4, 0.2],
            [0.5, 1, 0.5, 0.5, 7, 5],
            [1, 1, 1, 1, 1, 1],
            [3, 3, 3, 2, 1, 1],
        ],
        dtype=float,
    )
    base_hours = np.array([7, 17, 19, 19, 15, 14])
    base_lead = np.array([3, 10, 30, 36, 110, 180], dtype=float)
    rows: list[tuple] = []
    for resident_idx, count in enumerate(counts):
        persona = int(rng.choice(6, p=[0.28, 0.17, 0.22, 0.12, 0.09, 0.12]))
        weights = (
            personas[persona]
            * np.asarray(config.popularity)
            * rng.lognormal(0, 0.28, 6)
        )
        new_weights = np.roll(weights, int(rng.integers(1, 6)))
        changes = rng.random() < config.drift_fraction
        transition = rng.uniform(0.45, 0.75)
        lead_factor = rng.lognormal(0, 0.35)
        hour_shift = rng.normal(0, 0.6)
        # Some residents arrive late or become inactive. Most span the full window.
        arrival = (
            rng.integers(total_days // 2, int(total_days * 0.85))
            if resident_idx % 19 == 0
            else rng.integers(0, 28)
        )
        departure = (
            int(total_days * rng.uniform(0.7, 0.95))
            if resident_idx % 17 == 0
            else total_days
        )
        departure = max(departure, arrival + 14)
        cadence = (departure - arrival) / (count + 1)
        regular = rng.random() < 0.7
        if regular:
            days = (
                arrival
                + (np.arange(count) + 1) * cadence
                + rng.normal(0, max(1, cadence * 0.3), count)
            )
        else:
            days = rng.uniform(arrival, departure, count)
        for raw_day in np.sort(np.clip(days, arrival, departure - 1)):
            day = start + pd.Timedelta(days=int(raw_day))
            progress = raw_day / total_days
            w = weights.copy()
            if changes and progress > transition:
                w = 0.25 * w + 0.75 * new_weights
            w[1] *= 1.7 if day.month in [3, 4, 5, 6] else 0.65
            w[0] *= 1.4 if day.month in [1, 2] else 1
            if day.day_of_week >= 5:
                w[[1, 4, 5]] *= 1.8
            probability = (
                1 - config.exploration
            ) * w / w.sum() + config.exploration / 6
            facility_idx = int(rng.choice(6, p=probability))
            # Weekend swimmers and social residents shift most preferred sessions to weekends.
            if (persona == 1 and facility_idx == 1) or (
                persona == 3 and facility_idx >= 4
            ):
                if rng.random() < 0.8:
                    day += pd.Timedelta(days=(5 - day.day_of_week) % 7)
            hour = base_hours[facility_idx] + hour_shift
            if facility_idx == 0 and rng.random() < 0.18:
                hour = 18 + hour_shift
            if facility_idx == 1 and day.day_of_week >= 5:
                hour -= 2
            hour += rng.normal(0, 0.65 if regular else 1.5)
            if rng.random() < 0.035:
                hour = rng.uniform(6, 22)
            minutes = int(np.clip(np.round(hour * 4) * 15, 5 * 60, 22 * 60))
            usage = min(day.normalize(), end - pd.Timedelta(days=1)) + pd.Timedelta(
                minutes=minutes
            )
            lead_hours = float(
                np.clip(
                    rng.lognormal(np.log(base_lead[facility_idx] * lead_factor), 0.55),
                    0.25,
                    720,
                )
            )
            booked = usage - pd.Timedelta(seconds=int(lead_hours * 3600))
            # Exclude pre-window booking creation without moving scheduled usage beyond the window.
            if booked < start:
                booked = start + (usage - start) * 0.3
            rows.append(
                (
                    f"R-{resident_idx + 1:04d}",
                    FACILITIES[facility_idx],
                    booked.floor("s"),
                    usage,
                )
            )
    data = pd.DataFrame(
        rows,
        columns=[
            "resident_reference",
            "facility_reference",
            "booking_timestamp",
            "usage_timestamp",
        ],
    )
    data = data.sort_values(
        ["booking_timestamp", "resident_reference"], kind="stable"
    ).reset_index(drop=True)
    data.insert(0, "booking_id", [f"B-{i + 1:06d}" for i in range(len(data))])
    return data
