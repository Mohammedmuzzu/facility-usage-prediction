# Data dictionary

`data/facility_bookings.csv` has exactly these columns, sorted by booking creation timestamp. CSV dates contain explicit offsets and load into Asia/Kolkata-aware Pandas timestamps.

| Field | Type | Meaning / constraint |
|---|---|---|
| booking_id | string | Unique technical reference, e.g. B-000001 |
| resident_reference | string | Synthetic R-0001 through R-0500; identifier, not a model feature |
| facility_reference | string | Gym, Swimming Pool, Badminton Court, Tennis Court, Clubhouse or Multipurpose Hall |
| booking_timestamp | timezone-aware datetime | Booking creation; strictly before scheduled usage |
| usage_timestamp | timezone-aware datetime | Scheduled usage; does not establish attendance |

Allowed lead: (0, 720] hours, with generation constrained to [0.25, 720]. Booking IDs are assigned after global creation sorting. Four fields are business fields; booking_id is technical. No persona, true future preference, hidden activity rate or change-point label is exported.

`dataset_metadata.json` records seed/configuration, schema validation status, booking/usage ranges, population counts, class counts, resident-frequency quantiles, sparse population, lead-time quantiles, imbalance ratio and SHA-256 of the CSV bytes. `outputs/dataset_profile.json` mirrors the generation-time profile.

Sequential targets are created inside the pipeline: next booking ID/facility/creation/usage, prediction cutoff, elapsed usage days and advance lead hours. These target fields never join the feature matrix. Output prediction CSVs contain targets for review, predictions, unrounded errors, exact/tolerance booleans, counts out of four, activity group, seasonal transition and historical preference-shift diagnostics.

Excel stores local naive datetime values because Excel does not preserve timezone-aware datetimes. The Performance Summary explicitly states Asia/Kolkata. Display formatting does not alter raw evaluated errors. Native Excel datetime precision is lower than Pandas nanoseconds; full fidelity is retained in the CSV/JSON artifacts and evaluation occurs before export.
