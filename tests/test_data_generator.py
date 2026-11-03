from dataclasses import replace
import pandas as pd
from src.data_generator import generate_bookings
from src.data_validation import validate_bookings


def test_reproducible_and_seed_sensitive(config, bookings):
    pd.testing.assert_frame_equal(bookings, generate_bookings(config))
    assert not bookings.equals(generate_bookings(replace(config, seed=43)))


def test_schema_population_temporal_and_imbalance(config, bookings):
    metadata = validate_bookings(bookings, config)
    assert metadata["residents"] == config.residents
    assert metadata["bookings"] == config.bookings
    assert metadata["facilities"] == 6
    assert metadata["imbalance_ratio"] > 1.5
    assert (bookings.booking_timestamp < bookings.usage_timestamp).all()
    assert bookings.booking_timestamp.is_monotonic_increasing
    assert not any("persona" in column or "drift" in column for column in bookings)
