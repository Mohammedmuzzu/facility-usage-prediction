# Dataset profile

```json
{
  "validation_passed": true,
  "residents": 500,
  "facilities": 6,
  "bookings": 12500,
  "timezone": "Asia/Kolkata",
  "booking_range": [
    "2025-01-02 02:34:51+05:30",
    "2026-06-30 12:03:56+05:30"
  ],
  "usage_range": [
    "2025-01-03 19:15:00+05:30",
    "2026-06-30 15:30:00+05:30"
  ],
  "facility_distribution": {
    "Gym": 4181,
    "Swimming Pool": 2922,
    "Badminton Court": 2077,
    "Tennis Court": 1340,
    "Clubhouse": 1232,
    "Multipurpose Hall": 748
  },
  "resident_booking_frequency": {
    "count": 500.0,
    "mean": 25.0,
    "std": 21.39156839497267,
    "min": 1.0,
    "25%": 8.0,
    "50%": 20.0,
    "75%": 36.0,
    "max": 141.0
  },
  "lead_time_hours": {
    "count": 12500.0,
    "mean": 39.83248906666667,
    "std": 72.21505493588647,
    "min": 0.25,
    "10%": 2.105166666666667,
    "50%": 12.048194444444444,
    "90%": 111.49861111111112,
    "99%": 364.06663611111134,
    "max": 720.0
  },
  "residents_with_at_most_5_bookings": 97,
  "imbalance_ratio": 5.589572192513369,
  "generation_config": {
    "residents": 500,
    "bookings": 12500,
    "months": 18,
    "start": "2025-01-01",
    "seed": 42,
    "drift_fraction": 0.28,
    "exploration": 0.08,
    "sparse_fraction": 0.12,
    "popularity": [
      1.8,
      1.2,
      1.1,
      0.8,
      0.65,
      0.4
    ]
  },
  "seed": 42,
  "dataset_sha256": "294deeef05eb425ab6abba3ff2e715cc988297b85aa282c5ce0bdf2117d9ee4f"
}
```
