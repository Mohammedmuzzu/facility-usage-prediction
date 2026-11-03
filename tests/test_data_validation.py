import pandas as pd
import pytest
from src.data_validation import validate_bookings


@pytest.mark.parametrize(
    "issue",
    [
        "duplicate",
        "resident",
        "facility",
        "missing",
        "reversed",
        "too_long",
        "unordered",
        "naive",
    ],
)
def test_rejects_bad_data(bookings, config, issue):
    bad = bookings.copy()
    if issue == "duplicate":
        bad.loc[1, "booking_id"] = bad.loc[0, "booking_id"]
    elif issue == "resident":
        bad.loc[0, "resident_reference"] = "R-9999"
    elif issue == "facility":
        bad.loc[0, "facility_reference"] = "Unknown facility"
    elif issue == "missing":
        bad.loc[0, "facility_reference"] = None
    elif issue == "reversed":
        bad.loc[0, "usage_timestamp"] = bad.loc[0, "booking_timestamp"]
    elif issue == "too_long":
        bad.loc[0, "usage_timestamp"] = bad.loc[0, "booking_timestamp"] + pd.Timedelta(
            days=40
        )
    elif issue == "unordered":
        bad = bad.iloc[::-1]
    else:
        bad["booking_timestamp"] = bad.booking_timestamp.dt.tz_localize(None)
    with pytest.raises(ValueError):
        validate_bookings(bad, config)
