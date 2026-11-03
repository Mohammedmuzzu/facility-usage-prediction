# Historical feature definitions

For resident r and cutoff c, H contains only records for r with booking creation <= c, sorted by creation. A record's scheduled usage and lead are known from its creation onward. Features are constructed independently for every cutoff. Resident IDs, target fields and personas are excluded.

| Features | Formula / interpretation | Empty-history default |
|---|---|---|
| total_bookings, cold_start | len(H); indicator len(H)<3 | 0, 1 |
| days_since_last | (c - latest creation) / 86400 seconds | 0 |
| mean_interval, median_interval | Consecutive creation gaps, days | 14 days when <2 records |
| frequency_per_30d | len(H) * 30 / max(30, observation span in days) | 0 |
| regularity_cv | Population std of creation gaps / max(mean gap, 0.01) | 1 |
| bookings_7d, bookings_30d, bookings_90d | Count of H within each trailing elapsed window | 0 |
| facility_diversity | Number of distinct booked facilities | 0 |
| most_frequent_facility | Historical mode; ties use configured facility order | Unknown |
| most_recent_facility | Facility of latest known creation | Unknown |
| mean_lead_hours, median_lead_hours | Scheduled usage minus creation, hours | 24 hours |
| prediction_weekday/hour/month/season/weekend | Local cutoff calendar context; season=(month mod 12)//3 | Always known |
| previous_usage_weekday/hour | Latest known booking's scheduled usage context | -1 |
| preferred_weekday/hour | Historical scheduled usage mode; ties choose smallest | -1 / 12 |
| usage_sin, usage_cos | Mean sin/cos(2π * minute / 1440) across H | 0 / -1 (noon) |
| last_usage_offset_days | Latest known scheduled usage minus cutoff, days | 0 |
| facility_i_count | Count for facility i | 0 |
| facility_i_preference | (count_i + 1) / (len(H) + 6) | 1/6 |
| facility_i_recent_preference | (sum weight_i + 1) / (sum weights + 6), weight=exp(-age_days/60) | 1/6 |
| facility_i_lead_hours | Median lead for known facility i bookings | 24 hours |
| facility_i_usage_sin/cos | Facility-specific mean circular targets | 0 / -1 |
| weekday_d_preference | (weekday count_d + 1) / (len(H) + 7) | 1/7 |

Facility indices follow the order in `FACILITIES`: gym, swimming, badminton, tennis, clubhouse, multipurpose hall. The exact feature names are saved in `models/model_metadata.json`.

There is no learned feature imputer. Defaults are fixed beforehand. One-hot encoders are fitted only on training rows and ignore unknown categorical values. Cold-start inference (<2 known bookings) uses global training medians for timing and available resident facility history. The diagnostic `cold_start` feature uses <3 to expose a broader sparse regime; the fallback threshold is deliberately <2, where interval statistics do not exist.

`days_since_last` is normally zero for event-driven training cutoffs; inference at later cutoffs is supported but not independently validated as a scheduled daily prediction policy. Aggregates describe scheduled bookings, not completed facility visits.
