"""Reviewer UI: streamlit run app/streamlit_app.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
import streamlit as st
from src.config import ROOT
from src.predict import PredictionEngine
from src.utils import load_bookings, timestamp
from src.service_controls import ServiceSettings, authenticate, authorize
from fastapi import HTTPException

st.set_page_config(page_title="Facility Forecast", page_icon="🏘️", layout="wide")
st.title("Facility Forecast")
st.caption("Next observed booking · Asia/Kolkata · Synthetic community data")
try:
    settings = ServiceSettings.from_env()
    if settings.environment == "production":
        import os

        if os.getenv("FACILITY_DASHBOARD_TRUSTED_PROXY") != "true":
            st.error(
                "Production dashboard requires a private authenticated reverse proxy. Configure the deployment before exposing resident data."
            )
            st.stop()
    if settings.auth_file:
        operator_key = st.text_input(
            "Operator API key", type="password", key="operator_key"
        )
        try:
            authorize(authenticate(settings, operator_key))
        except HTTPException:
            st.info(
                "An authorized administrator key is required to view community data."
            )
            st.stop()
except (ValueError, OSError, KeyError):
    st.error("Service security configuration is unavailable. Contact the operator.")
    st.stop()
if not (ROOT / "outputs/evaluation_metrics.json").exists():
    st.info("Run python scripts/run_pipeline.py to generate the review artifacts.")
    st.stop()


@st.cache_resource
def resources(model_mtime: float, data_mtime: float):
    data = load_bookings(ROOT / "data/facility_bookings.csv")
    return data, PredictionEngine(ROOT / "models/model_bundle.joblib", data)


data, engine = resources(
    (ROOT / "models/model_bundle.joblib").stat().st_mtime,
    (ROOT / "data/facility_bookings.csv").stat().st_mtime,
)
metrics = json.loads((ROOT / "outputs/evaluation_metrics.json").read_text())
review = pd.read_csv(ROOT / "outputs/test_predictions.csv")
overview, resident_tab, review_tab, evaluation_tab = st.tabs(
    ["Overview", "Resident Prediction", "Prediction Review", "Model Evaluation"]
)
with overview:
    a, b, c, d = st.columns(4)
    a.metric("Residents", data.resident_reference.nunique())
    b.metric("Bookings", len(data))
    primary = metrics.get("champion", metrics["ml"])
    c.metric("Facility accuracy", f"{primary['facility']['accuracy']:.1%}")
    d.metric("Tolerance 4/4", f"{primary['overall']['tolerance_4_of_4_rate']:.1%}")
    st.bar_chart(data.facility_reference.value_counts())
    st.warning(
        "Synthetic evaluation does not establish real-world performance. Notification time is a booking-time proxy, with no measured engagement or conversion benefit."
    )
with resident_tab:
    resident = st.selectbox("Resident", sorted(data.resident_reference.unique()))
    default_cutoff = (data.booking_timestamp.max() + pd.Timedelta(days=1)).isoformat()
    cutoff_text = st.text_input(
        "Prediction cutoff (ISO timestamp with timezone)", default_cutoff
    )
    if st.button("Predict next booking", type="primary"):
        try:
            result = engine.predict(resident, cutoff_text)
            st.json(result)
            cutoff = timestamp(cutoff_text)
            history = engine.builder.history(resident, cutoff)
            st.subheader("History available at this cutoff")
            st.dataframe(history, width="stretch")
            if len(history):
                st.bar_chart(history.facility_reference.value_counts(normalize=True))
        except ValueError as exc:
            st.error(str(exc))
with review_tab:
    selected_resident = st.selectbox(
        "Filter resident", ["All", *sorted(review.resident_reference.unique())]
    )
    selected_outcome = st.selectbox(
        "Filter outcome", ["All", "Correct", "Within tolerance", "Incorrect"]
    )
    selected = review.copy()
    if selected_resident != "All":
        selected = selected[selected.resident_reference == selected_resident]
    if selected_outcome != "All":
        selected = selected[selected.overall_result == selected_outcome]
    st.dataframe(selected, width="stretch")
    st.download_button(
        "Download Excel review",
        (ROOT / "outputs/facility_prediction_review.xlsx").read_bytes(),
        "facility_prediction_review.xlsx",
    )
with evaluation_tab:
    rows = []
    for name, values in metrics.items():
        rows.append(
            {
                "Model": name,
                "Facility accuracy": values["facility"]["accuracy"],
                "Macro F1": values["facility"]["macro_f1"],
                "Days MAE": values["usage_day"]["mae_days"],
                "Circular minutes MAE": values["usage_hour"]["circular_mae_minutes"],
                "Notification hours MAE": values["notification"]["mae_hours"],
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")
    st.image(str(ROOT / "outputs/figures/facility_confusion_matrix.png"))
    st.image(str(ROOT / "outputs/figures/prediction_error_distributions.png"))
