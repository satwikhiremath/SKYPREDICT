import json
from pathlib import Path
from datetime import time

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from common import load_model_bundle, build_prediction_input


# =========================================================
# CONFIGURATION
# =========================================================

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models" / "flight_delay_model.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

ANALYTICS_PATH = BASE / "data" / "analytics"

st.set_page_config(
    page_title="SKYPREDICT | Flight Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# DESIGN
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f4f7fc;
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
    color: #aabbd5 !important;
    font-size: 11px;
    letter-spacing: 2px;
}

.hero {
    background: linear-gradient(115deg, #10213e, #1769aa);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
}

.hero p {
    color: #dcecff;
    font-size: 15px;
}

.section-title {
    color: #172b4d;
    font-size: 23px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    border: 1px solid #e7ebf2;
    border-radius: 15px;
    padding: 20px;
    min-height: 120px;
}

.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

.metric-value {
    color: #1769aa;
    font-size: 27px;
    font-weight: 800;
    margin-top: 10px;
}

.panel {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e7ebf2;
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(135deg, #10213e, #1769aa);
    padding: 30px;
    border-radius: 18px;
    color: white;
    text-align: center;
}

.result-card h2 {
    color: white;
}

.footer {
    text-align: center;
    color: #8793a5;
    padding: 30px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

bundle = None
pipeline = None
model_error = None

if MODEL_PATH.exists():

    try:
        bundle = load_model_bundle(MODEL_PATH)
        pipeline = bundle["pipeline"]

    except Exception as exc:
        model_error = str(exc)

else:
    model_error = "Trained model file not found."


# =========================================================
# LOAD METRICS
# =========================================================

metrics = {}

if METRICS_PATH.exists():

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

    except Exception as exc:
        st.warning(f"Could not load evaluation metrics: {exc}")


# =========================================================
# LOAD ANALYTICS DATA
# =========================================================

@st.cache_data
def load_csv(filename):

    path = ANALYTICS_PATH / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_overview():

    path = ANALYTICS_PATH / "overview_stats.json"

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return {}


airline_df = load_csv("airline_stats.csv")
origin_df = load_csv("origin_airport_stats.csv")
destination_df = load_csv("destination_airport_stats.csv")
monthly_df = load_csv("monthly_stats.csv")
yearly_df = load_csv("yearly_stats.csv")
route_df = load_csv("top_route_stats.csv")

overview = load_overview()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_number(value):

    return f"{int(value):,}"


def format_percentage(value):

    return f"{float(value):.2f}%"


def format_metric(key):

    value = metrics.get(key)

    if value is None:
        return "N/A"

    if isinstance(value, (int, float)):

        if key in ["accuracy", "precision", "recall", "f1",
                   "roc_auc", "best_cv_f1"]:

            return f"{value * 100:.2f}%"

        return f"{value:,.2f}"

    return str(value)


def show_hero(title, description, tag):

    st.markdown(f"""
    <div class="hero">

        <div style="font-size:11px;letter-spacing:2px;">
            {tag}
        </div>

        <h1>{title}</h1>

        <p>{description}</p>

    </div>
    """, unsafe_allow_html=True)


def show_metric(label, value):

    st.markdown(f"""
    <div class="metric-card">

        <div class="metric-label">
            {label}
        </div>

        <div class="metric-value">
            {value}
        </div>

    </div>
    """, unsafe_allow_html=True)


def show_chart(data, x, y, title, chart_type="bar"):

    if data.empty:
        st.info(f"{title}: Data is not available.")
        return

    if x not in data.columns or y not in data.columns:
        st.warning(f"{title}: Required columns are missing.")
        return

    if chart_type == "line":

        fig = px.line(
            data,
            x=x,
            y=y,
            markers=True,
            title=title
        )

    else:

        fig = px.bar(
            data,
            x=x,
            y=y,
            title=title
        )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def show_data(data):

    if not data.empty:
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No data available.")


# =========================================================
# SIDEBAR
# =========================================================

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
            "Flight Insights",
            "Model Performance",
            "About"
        ]
    )

    st.divider()

    if pipeline is not None:
        st.success("Prediction Engine Online")
    else:
        st.error("Prediction Engine Offline")

    if overview:
        st.caption(
            f"Historical Flights: "
            f"{format_number(overview.get('total_flights', 0))}"
        )

    st.markdown(
        '<p class="footer">SKYPREDICT v2.0</p>',
        unsafe_allow_html=True
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    show_hero(
        "Know Before You Fly.",
        "Intelligent flight analytics and machine-learning-based "
        "flight delay predictions.",
        "FLIGHT INTELLIGENCE PLATFORM"
    )

    st.markdown(
        '<div class="section-title">Your Dashboard</div>',
        unsafe_allow_html=True
    )

    total = overview.get("total_flights", 0)
    delayed = overview.get("delayed_flights", 0)
    not_delayed = overview.get("not_delayed_flights", 0)
    delay_rate = overview.get("delay_rate_pct", 0)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_metric("Total Flights", format_number(total))

    with c2:
        show_metric("Delayed Flights", format_number(delayed))

    with c3:
        show_metric("On-Time Flights", format_number(not_delayed))

    with c4:
        show_metric("Delay Percentage", format_percentage(delay_rate))

    st.markdown(
        '<div class="section-title">Historical Flight Trends</div>',
        unsafe_allow_html=True
    )

    if not yearly_df.empty:

        show_chart(
            yearly_df,
            "YEAR",
            "flights",
            "Yearly Flight Volume",
            "line"
        )

    if not monthly_df.empty:

        monthly_df["period"] = (
            monthly_df["YEAR"].astype(str)
            + "-"
            + monthly_df["MONTH"].astype(str).str.zfill(2)
        )

        show_chart(
            monthly_df,
            "period",
            "flights",
            "Monthly Flight Volume",
            "line"
        )

    st.markdown(
        '<div class="section-title">Prediction Model Snapshot</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.metric("Accuracy", format_metric("accuracy"))

    with b:
        st.metric("Precision", format_metric("precision"))

    with c:
        st.metric("Recall", format_metric("recall"))


# =========================================================
# PREDICT FLIGHT
# =========================================================

elif page == "Predict Flight":

    show_hero(
        "Check Your Flight Risk.",
        "Enter flight information to estimate the possibility "
        "of a departure delay.",
        "PREDICTION STUDIO"
    )

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

    col1, col2 = st.columns(2)

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

    predict_button = st.button(
        "✈ PREDICT FLIGHT DELAY",
        use_container_width=True
    )

    if predict_button:

        if origin == destination:

            st.error(
                "Departure and arrival airports must be different."
            )

        elif pipeline is None or bundle is None:

            st.error(
                "Prediction model is unavailable. "
                "Check the model file and common.py."
            )

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

                    probabilities = pipeline.predict_proba(
                        input_data
                    )[0]

                    classes = list(pipeline.classes_)

                    if 1 in classes:

                        probability = float(
                            probabilities[classes.index(1)]
                        )

                risk = (
                    f"{probability:.1%}"
                    if probability is not None
                    else "Unavailable"
                )

                if int(prediction) == 1:

                    st.markdown(f"""
                    <div class="result-card">

                        <h2>⚠ Flight Delay Likely</h2>

                        <p>Your flight is classified as likely to be delayed.</p>

                        <h1>{risk}</h1>

                        <p>Estimated delay probability</p>

                    </div>
                    """, unsafe_allow_html=True)

                    st.warning(
                        "Consider allowing extra time for your journey. "
                        "This is a machine-learning estimate, not a live "
                        "airline update."
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
                        "The model predicts a lower risk of delay. "
                        "Confirm your flight status with the airline."
                    )

                st.caption(
                    f"Route: {origin} → {destination} | "
                    f"Airline: {airline_name} | "
                    f"Departure: {departure_time.strftime('%H:%M')}"
                )

            except Exception as exc:

                st.error(f"Prediction failed: {exc}")


# =========================================================
# FLIGHT INSIGHTS
# =========================================================

elif page == "Flight Insights":

    show_hero(
        "Explore Flight Insights.",
        "Understand historical flight patterns, airline performance "
        "and airport delay statistics.",
        "HISTORICAL FLIGHT ANALYTICS"
    )

    st.markdown(
        '<div class="section-title">Airline Analysis</div>',
        unsafe_allow_html=True
    )

    if not airline_df.empty:

        c1, c2 = st.columns(2)

        with c1:

            show_chart(
                airline_df.sort_values(
                    "flights",
                    ascending=False
                ),
                "OP_CARRIER",
                "flights",
                "Flight Volume by Airline"
            )

        with c2:

            show_chart(
                airline_df.sort_values(
                    "delay_rate_pct",
                    ascending=False
                ),
                "OP_CARRIER",
                "delay_rate_pct",
                "Airline Delay Percentage"
            )

        st.markdown("### Airline Statistics")

        show_data(airline_df)

    else:

        st.warning("Airline analytics file not found.")

    # AIRPORT ANALYSIS

    st.markdown(
        '<div class="section-title">Airport Analysis</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        show_chart(
            origin_df.head(15),
            "ORIGIN",
            "flights",
            "Top Departure Airports"
        )

    with c2:

        show_chart(
            destination_df.head(15),
            "DEST",
            "flights",
            "Top Arrival Airports"
        )

    # MONTHLY ANALYSIS

    st.markdown(
        '<div class="section-title">Monthly Flight Trends</div>',
        unsafe_allow_html=True
    )

    if not monthly_df.empty:

        monthly_df = monthly_df.copy()

        monthly_df["period"] = (
            monthly_df["YEAR"].astype(str)
            + "-"
            + monthly_df["MONTH"].astype(str).str.zfill(2)
        )

        c1, c2 = st.columns(2)

        with c1:

            show_chart(
                monthly_df,
                "period",
                "flights",
                "Monthly Flight Volume",
                "line"
            )

        with c2:

            show_chart(
                monthly_df,
                "period",
                "delay_rate_pct",
                "Monthly Delay Percentage",
                "line"
            )

        show_data(monthly_df)

    # YEARLY ANALYSIS

    st.markdown(
        '<div class="section-title">Yearly Flight Trends</div>',
        unsafe_allow_html=True
    )

    if not yearly_df.empty:

        c1, c2 = st.columns(2)

        with c1:

            show_chart(
                yearly_df,
                "YEAR",
                "flights",
                "Yearly Flight Volume",
                "line"
            )

        with c2:

            show_chart(
                yearly_df,
                "YEAR",
                "delay_rate_pct",
                "Yearly Delay Percentage",
                "line"
            )

    # ROUTE ANALYSIS

    st.markdown(
        '<div class="section-title">Popular Flight Routes</div>',
        unsafe_allow_html=True
    )

    if not route_df.empty:

        show_chart(
            route_df.head(15),
            "ROUTE",
            "flights",
            "Top Flight Routes"
        )

        show_data(route_df)

    else:

        st.info("Route analytics are not available.")


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    show_hero(
        "Prediction Performance.",
        "Evaluation results of the trained Logistic Regression model.",
        "MODEL INTELLIGENCE"
    )

    if metrics:

        st.markdown("### Evaluation Metrics")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Accuracy", format_metric("accuracy"))

        with c2:
            st.metric("Precision", format_metric("precision"))

        with c3:
            st.metric("Recall", format_metric("recall"))

        with c4:
            st.metric("F1 Score", format_metric("f1"))

        c5, c6 = st.columns(2)

        with c5:
            st.metric("ROC-AUC", format_metric("roc_auc"))

        with c6:
            st.metric("Cross Validation F1", format_metric("best_cv_f1"))

        st.markdown("### Model Information")

        st.write(
            "Selected Model:",
            metrics.get("model", "Not available")
        )

        st.write(
            "Training Records:",
            format_number(metrics.get("training_rows", 0))
        )

        st.write(
            "Testing Records:",
            format_number(metrics.get("testing_rows", 0))
        )

        st.write(
            "Training Period:",
            metrics.get("training_years", "Not available")
        )

        st.write(
            "Testing Period:",
            metrics.get("testing_years", "Not available")
        )

        st.info(
            "These results are based on historical model evaluation. "
            "Predictions are estimates and are not live airline information."
        )

    else:

        st.warning(
            "Evaluation metrics are unavailable. "
            "Check models/metrics.json."
        )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    show_hero(
        "About SKYPREDICT",
        "Know More. Travel Smarter.",
        "ABOUT OUR PLATFORM"
    )

    st.markdown("""
    ### Welcome to SKYPREDICT

    SKYPREDICT is a flight intelligence platform that combines
    historical flight analytics and machine-learning-based
    flight delay prediction.

    ### Main Features

    - Flight delay prediction.
    - Airline performance analysis.
    - Airport-level flight statistics.
    - Monthly and yearly flight trends.
    - Popular flight route analysis.
    - Machine learning model evaluation.

    ### Machine Learning

    Our project uses a Logistic Regression classification model
    to estimate whether a flight is likely to experience a
    departure delay exceeding 15 minutes.

    ### Dataset

    Historical airline flight records from 2009–2018 were used
    for model development and analytics.

    ### Our Vision

    To make air travel more informed through historical
    flight intelligence and machine learning.
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    SKYPREDICT · Intelligent Flight Analytics

    <br>

    Machine-learning-based estimates.
    Not live airline information.

</div>
""", unsafe_allow_html=True)
