"""Validation-only permutation importance: diagnostic association, not causation."""

import pandas as pd
from sklearn.inspection import permutation_importance


def facility_importance(bundle: dict, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    # Keep this optional diagnostic small and away from final-test model selection.
    result = permutation_importance(
        bundle["models"]["facility"],
        X,
        y.target_facility,
        scoring="f1_macro",
        n_repeats=2,
        random_state=42,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "validation_macro_f1_decrease": result.importances_mean,
            "standard_deviation": result.importances_std,
        }
    ).sort_values("validation_macro_f1_decrease", ascending=False)
