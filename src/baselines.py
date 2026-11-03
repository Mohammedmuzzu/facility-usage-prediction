"""Train-only global heuristics and cutoff-limited resident-history heuristics."""
import numpy as np
import pandas as pd
from src.config import FACILITIES
from src.utils import minutes_of_day


class Baselines:
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "Baselines":
        self.facility = y.target_facility.mode().iloc[0]
        self.days = float(y.days_until_usage.median())
        self.lead = float(y.lead_hours.median())
        self.minute = float(pd.Series(minutes_of_day(y.target_usage_timestamp)).mode().iloc[0])
        return self

    def predict(self, X: pd.DataFrame, kind: str = "resident") -> dict:
        n = len(X)
        if kind == "global":
            return {"facility": np.repeat(self.facility, n), "days": np.repeat(self.days, n),
                    "minutes": np.repeat(self.minute, n), "lead_hours": np.repeat(self.lead, n)}
        known = X.total_bookings.to_numpy() >= 2
        # The typical next creation interval plus a resident's typical lead approximates next usage.
        return {"facility": np.where(X.total_bookings > 0, X.most_frequent_facility, self.facility),
                "days": np.where(known, X.median_interval + X.median_lead_hours / 24, self.days),
                "minutes": np.where(known, X.preferred_hour * 60, self.minute),
                "lead_hours": np.where(known, X.median_lead_hours, self.lead)}
