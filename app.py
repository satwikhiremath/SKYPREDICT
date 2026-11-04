import json
from pathlib import Path
from datetime import time

import pandas as pd
import streamlit as st

from common import load_model_bundle, build_prediction_input


# ==================================================
# CONFIGURATION
# ==================================================

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models" / "flight_delay_model.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

st.set_page_config(
    page_title="SKYPREDICT | Flight Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# DESIGN
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #101d35;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.brand {
    font-size: 27px;
    font-weight: 800;
    color: white;
    letter-spacing: 1px;
}

.brand span {
    color: #55b5ff;
}

.sidebar-caption {
    color: #9aacc9 !important;
    font-size: 12px;
    letter-spacing: 2px;
    margin-top: 5px;
}

.hero {
    background: linear-gradient(115deg, #10213e, #174d83);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #d5e7ff;
    font-size: 15px;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 11px;
    letter-spacing: 1px;
    margin-bottom: 15px;
}

.section-title {
    color: #172b4d;
    font-size: 23px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    border: 1px solid #e7ebf2;
    border-radius: 16px;
    padding: 22px;
    min-height: 125px;
    box-shadow: 0 4px 15px rgba(20,40,80,0.035);
}

.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    color: #142b4d;
    font-size: 28px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-note {
    color: #8a96a8;
    font-size: 12px;
    margin-top: 5px;
}

.panel {
    background: white;
    border: 1px solid #e7ebf2;
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(135deg, #10213e, #174d83);
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
}

.result-card h2 {
    color: white;
    font-size: 28px;
}

.result-card p {
    color: #d7e9ff;
}

.small-muted {
    color: #8290a5;
    font-size: 12px;
}

div.stButton > button {
    background: #1769aa;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 22px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #0d4d83;
    color: white;
}

.stSelectbox label,
.stTimeInput label {
    font-weight: 600;
    color: #253b5b;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL AND METRICS
# ==================================================

bundle = None
pipeline = None
metrics = {}

if MODEL_PATH.exists():
    try:
        bundle = load_model_bundle(MODEL_PATH)
        pipeline = bundle["pipeline"]
    except Exception as exc:
        st.error(f"Unable to load prediction engine: {exc}")

if METRICS_PATH.exists():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)
    except Exception:
        metrics = {}


def get_metric(*keys):

    for key in keys:
        if key in metrics:
            value = metrics[key]

            if isinstance(value, (int, float)):
                return f"{value:.2%}" if value <= 1 else f"{value:.2f}"

            return str(value)

    return "—"


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">SKY<span>PREDICT</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-caption">FLIGHT INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Flight",
            "Model Performance",
            "About"
        ]
    )

    st.divider()

    if pipeline is not None:
        st.success("Prediction engine online")
    else:
        st.warning("Prediction engine offline")

    st.markdown(
        '<p class="small-muted">SKYPREDICT v1.0</p>',
        unsafe_allow_html=True
    )


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">✈ INTELLIGENT FLIGHT ANALYTICS</div>
        <h1>Know Before You Fly.</h1>
        <p>
        Predict flight delays, understand travel patterns,
        and make smarter flight decisions with machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Your Flight Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Platform</div>
            <div class="metric-value">SKYPREDICT</div>
            <div class="metric-note">Flight intelligence</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Prediction</div>
            <div class="metric-value">ML Based</div>
            <div class="metric-note">Flight delay classification</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        status = "Online" if pipeline is not None else "Offline"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Engine Status</div>
            <div class="metric-value">{status}</div>
            <div class="metric-note">Local prediction service</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Prediction Inputs</div>
            <div class="metric-value">04</div>
            <div class="metric-note">Flight information fields</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### ✈ Start a Flight Prediction")
        st.write(
            "Enter your flight details to estimate whether "
            "your departure is likely to be delayed."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 📊 Model Performance")
        st.write(
            "Explore the evaluation results of the "
            "flight delay prediction model."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Model Snapshot</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.metric("Accuracy", get_metric("accuracy"))

    with b:
        st.metric("Precision", get_metric("precision"))

    with c:
        st.metric("Recall", get_metric("recall"))


# ==================================================
# PREDICT FLIGHT
# ==================================================

elif page == "Predict Flight":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">PREDICTION STUDIO</div>
        <h1>Check Your Flight Risk.</h1>
        <p>
        Enter your scheduled flight details and receive
        a machine-learning-based delay prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Flight Information</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    airlines = {
        "Southwest Airlines": "WN",
        "American Airlines": "AA",
        "Delta Air Lines": "DL",
        "United Airlines": "UA",
        "JetBlue Airways": "B6",
        "Alaska Airlines": "AS",
        "Spirit Airlines": "NK",
        "Frontier Airlines": "F9",
        "Other": "Other"
    }

    with col1:

        airline_name = st.selectbox(
            "Select Airline",
            list(airlines.keys())
        )

        origin = st.selectbox(
            "Departure Airport",
            ["JFK", "LAX", "ORD", "ATL", "DFW",
             "SFO", "DEN", "LAS", "MCO", "SEA"]
        )

    with col2:

        destination = st.selectbox(
            "Arrival Airport",
            ["LAX", "JFK", "ORD", "ATL", "DFW",
             "SFO", "DEN", "LAS", "MCO", "SEA"]
        )

        departure_time = st.time_input(
            "Scheduled Departure Time",
            value=time(13, 25)
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button(
        "✈ PREDICT FLIGHT DELAY",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if predict_button:

        if origin == destination:
            st.error("Departure and arrival airports must be different.")

        elif pipeline is None or bundle is None:
            st.error("The prediction engine is currently unavailable.")

        else:
            try:

                carrier = airlines[airline_name]

                departure_hhmm = (
                    departure_time.hour * 100
                    + departure_time.minute
                )

                input_data = build_prediction_input(
                    bundle,
                    carrier,
                    origin,
                    destination,
                    departure_hhmm
                )

                prediction = pipeline.predict(input_data)[0]

                probability = None

                if hasattr(pipeline, "predict_proba"):

                    probabilities = pipeline.predict_proba(input_data)[0]
                    classes = list(pipeline.classes_)

                    if 1 in classes:
                        probability = float(
                            probabilities[classes.index(1)]
                        )

                st.markdown("<br>", unsafe_allow_html=True)

                risk = (
                    f"{probability:.1%}"
                    if probability is not None else "Not available"
                )

                if int(prediction) == 1:

                    st.markdown(f"""
                    <div class="result-card">
                        <h2>⚠ Delay Likely</h2>
                        <p>Your flight is classified as likely to be delayed.</p>
                        <h1>{risk}</h1>
                        <p>Estimated delay probability</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.warning(
                        "Consider allowing extra time for your journey. "
                        "This is a model estimate, not a live airline update."
                    )

                else:

                    st.markdown(f"""
                    <div class="result-card">
                        <h2>✓ Delay Not Expected</h2>
                        <p>Your flight is classified as unlikely to be delayed.</p>
                        <h1>{risk}</h1>
                        <p>Estimated delay probability</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.success(
                        "The model predicts a lower risk of departure delay. "
                        "Always confirm your flight status with the airline."
                    )

                st.caption(
                    f"Route: {origin} → {destination} | "
                    f"Airline: {airline_name} | "
                    f"Departure: {departure_time.strftime('%H:%M')}"
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# ==================================================
# MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">MODEL INTELLIGENCE</div>
        <h1>Prediction Performance.</h1>
        <p>
        Monitor the evaluation results of the
        flight delay prediction engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not metrics:

        st.warning("Performance information is not available yet.")

    else:

        st.markdown(
            '<div class="section-title">Evaluation Metrics</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        metric_items = [
            ("Accuracy", ("accuracy",)),
            ("Precision", ("precision",)),
            ("Recall", ("recall",)),
            ("F1 Score", ("f1", "f1_score"))
        ]

        for column, (label, keys) in zip(
            [c1, c2, c3, c4],
            metric_items
        ):
            with column:
                st.metric(label, get_metric(*keys))

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown("### Evaluation Details")

        metric_df = pd.DataFrame([
            {"Metric": key, "Value": value}
            for key, value in metrics.items()
            if isinstance(value, (int, float, str))
        ])

        if not metric_df.empty:
            st.dataframe(
                metric_df,
                use_container_width=True,
                hide_index=True
            )

        st.markdown('</div>', unsafe_allow_html=True)


# ==================================================
# ABOUT
# ==================================================

elif page == "About":

    st.markdown("""
    <div class="hero">
        <h1>About SKYPREDICT</h1>
        <p>Know More. Travel Smarter.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to SKYPREDICT

    SKYPREDICT brings flight intelligence and delay
    prediction together in one simple platform.

    Our platform helps travellers explore flight
    information, understand delay patterns and make
    more informed travel decisions.

    ### What You Can Do

    - Predict possible flight delays.
    - Explore model performance.
    - Understand historical flight patterns.
    - Make more informed travel decisions.

    ### Our Vision

    To make air travel more informed, convenient
    and predictable through intelligent flight insights.

    ### Project Team

    - Satwik Hiremath
    - Nitish Yadav

    ### Project Title

    Flight Delay Prediction and Disruption Risk
    Classification System.
    """)


# ==================================================
# FOOTER
# ==================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#8995a7;font-size:12px;">
    SKYPREDICT · Intelligent Flight Analytics
    <br><br>
    Machine-learning-based estimates.
    Not live airline information.
</div>
""", unsafe_allow_html=True)
