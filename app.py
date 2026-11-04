import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import time, date

# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SKYPREDICT",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "flight_delay_model.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

# ==========================================
# DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
}

.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background-color: #10213e;
}

[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    background: linear-gradient(120deg, #10213e, #1769aa);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 40px;
    font-weight: 800;
}

.hero p {
    color: #dcecff;
    font-size: 16px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e5eaf2;
    margin-bottom: 15px;
}

.card h3 {
    color: #10213e;
    font-size: 15px;
}

.card h2 {
    color: #1769aa;
    font-size: 27px;
}

.section {
    color: #10213e;
    font-size: 24px;
    font-weight: bold;
    margin: 25px 0 15px 0;
}

.footer {
    text-align: center;
    color: #8793a5;
    padding: 30px;
    font-size: 12px;
}

.stButton button {
    background: #1769aa;
    color: white;
    border-radius: 9px;
    border: none;
    min-height: 45px;
    font-weight: bold;
}

.stButton button:hover {
    background: #10213e;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL INFORMATION
# ==========================================

metrics = {}

if METRICS_PATH.exists():
    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}

model_available = MODEL_PATH.exists()


def show_metric(label, value, description):
    st.markdown(f"""
    <div class="card">
        <h3>{label}</h3>
        <h2>{value}</h2>
        <p style="color:#8793a5;font-size:12px;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)


def metric_value(*keys):
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return f"{value:.2%}" if value <= 1 else f"{value:.2f}"
    return "Not available"


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("""
    <h1 style="font-size:27px;">
        SKY<span style="color:#55b5ff;">PREDICT</span>
    </h1>
    <p style="color:#aabbd5;font-size:11px;letter-spacing:2px;">
        FLIGHT INTELLIGENCE
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Predict Flight",
            "Flight Analytics",
            "Model Performance",
            "About"
        ]
    )

    st.divider()

    if model_available:
        st.success("Prediction engine available")
    else:
        st.warning("Prediction model not installed")


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Know Before You Fly.</h1>
        <p>
        Intelligent flight analytics and machine-learning-based
        flight delay predictions.
        </p>
        <p>✈ FLIGHT INTELLIGENCE PLATFORM</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section">Your Dashboard</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_metric(
            "PLATFORM",
            "SKYPREDICT",
            "Flight intelligence"
        )

    with c2:
        show_metric(
            "PREDICTION",
            "ML Based",
            "Flight delay classification"
        )

    with c3:
        show_metric(
            "MODEL STATUS",
            "ONLINE" if model_available else "OFFLINE",
            "Local prediction engine"
        )

    with c4:
        show_metric(
            "INPUT FIELDS",
            "09",
            "Model input features"
        )

    st.markdown(
        '<div class="section">Explore SKYPREDICT</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <h2>✈️</h2>
            <h3>Flight Prediction</h3>
            <p>Estimate the possibility of flight delays.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Prediction Studio", key="prediction_home"):
            st.session_state["navigation"] = "Predict Flight"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card">
            <h2>📊</h2>
            <h3>Flight Analytics</h3>
            <p>Explore flight patterns and model insights.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Explore Analytics", key="analytics_home"):
            st.session_state["navigation"] = "Flight Analytics"
            st.rerun()


# ==========================================
# FLIGHT PREDICTION
# ==========================================

elif page == "Predict Flight":

    from common import (
        load_model_bundle,
        build_prediction_input,
        predict_flight,
        predict_flight_probability
    )

    st.markdown("""
    <div class="hero">
        <h1>Flight Prediction</h1>
        <p>Enter your flight information to estimate delay risk.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section">Enter Flight Details</div>',
        unsafe_allow_html=True
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

    airports = [
        "JFK", "LAX", "ORD", "ATL", "DFW",
        "SFO", "DEN", "LAS", "MCO", "SEA"
    ]

    with st.form("flight_form"):

        col1, col2 = st.columns(2)

        with col1:
            airline = st.selectbox(
                "Select Airline",
                list(airlines.keys())
            )

            origin = st.selectbox(
                "Departure Airport",
                airports
            )

            departure = st.time_input(
                "Scheduled Departure Time",
                value=time(13, 25)
            )

        with col2:
            destination_options = [a for a in airports if a != origin]
            destination = st.selectbox(
                "Arrival Airport",
                destination_options
            )

            flight_date = st.date_input(
                "Flight Date",
                value=date(2018, 1, 1),
                min_value=date(2009, 1, 1),
                max_value=date(2018, 12, 31)
            )

        submitted = st.form_submit_button(
            "PREDICT FLIGHT DELAY",
            use_container_width=True
        )

    if submitted:

        if not model_available:
            st.error("Prediction model is not available. Check the models folder.")

        else:
            try:
                bundle = load_model_bundle(MODEL_PATH)

                departure_hhmm = (
                    departure.hour * 100 + departure.minute
                )

                X = build_prediction_input(
                    bundle,
                    airlines[airline],
                    origin,
                    destination,
                    departure_hhmm,
                    flight_date
                )

                prediction = predict_flight(bundle, X)[0]
                delay_probability = predict_flight_probability(bundle, X)

                st.divider()
                st.subheader("Your Flight Prediction")

                if int(prediction) == 1:
                    st.error("Flight Delay Likely")
                else:
                    st.success("Flight Delay Not Expected")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric(
                        "Estimated Delay Probability",
                        f"{delay_probability:.1%}"
                    )

                with col_b:
                    st.metric(
                        "Prediction Classification",
                        "Delayed" if int(prediction) == 1 else "Not Delayed"
                    )

                st.progress(float(delay_probability))

                st.info(
                    f"**Airline:** {airline}  |  "
                    f"**Route:** {origin} → {destination}  |  "
                    f"**Departure:** {departure.strftime('%H:%M')}  |  "
                    f"**Date:** {flight_date}"
                )

                with st.expander("View the nine model input features"):
                    st.dataframe(
                        X,
                        use_container_width=True,
                        hide_index=True
                    )

                st.caption(
                    "This is a historical machine-learning estimate, "
                    "not a live airline status update. The model was trained "
                    "using historical data from 2009–2018, so results for "
                    "other periods are not validated."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ==========================================
# FLIGHT ANALYTICS
# ==========================================

elif page == "Flight Analytics":

    st.markdown("""
    <div class="hero">
        <h1>Flight Analytics</h1>
        <p>Explore historical flight data from 2009 to 2018.</p>
    </div>
    """, unsafe_allow_html=True)

    DATA_PATH = BASE / "data" / "processed" / "flight_data.csv"

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")

    else:
        st.success("Historical flight dataset connected successfully.")

        @st.cache_data(show_spinner=False)
        def load_flight_analytics(file_path, modified_time):
            columns = [
                "YEAR",
                "OP_CARRIER",
                "ORIGIN",
                "DEST",
                "SEASON",
                "target"
            ]

            yearly = []
            airlines = []
            airports = []
            seasons = []

            total_flights = 0
            delayed_flights = 0

            chunk_size = 500_000

            for chunk in pd.read_csv(
                file_path,
                usecols=columns,
                chunksize=chunk_size
            ):
                chunk["target"] = pd.to_numeric(
                    chunk["target"],
                    errors="coerce"
                ).fillna(0)

                total_flights += len(chunk)
                delayed_flights += int(chunk["target"].sum())

                yearly.append(
                    chunk.groupby("YEAR")["target"]
                    .agg(["count", "sum"])
                )

                airlines.append(
                    chunk.groupby("OP_CARRIER")["target"]
                    .agg(["count", "sum"])
                )

                airports.append(
                    chunk.groupby("ORIGIN")["target"]
                    .agg(["count", "sum"])
                )

                seasons.append(
                    chunk.groupby("SEASON")["target"]
                    .agg(["count", "sum"])
                )

            def combine_results(data):
                result = pd.concat(data).groupby(level=0).sum()
                result["delay_rate"] = (
                    result["sum"] / result["count"] * 100
                )
                return result.reset_index()

            yearly_data = combine_results(yearly)
            airline_data = combine_results(airlines)
            airport_data = combine_results(airports)
            season_data = combine_results(seasons)

            return (
                total_flights,
                delayed_flights,
                yearly_data,
                airline_data,
                airport_data,
                season_data
            )

        if st.button("Load Historical Analytics"):
            with st.spinner(
                "Analyzing the historical flight dataset. "
                "The first run may take several minutes..."
            ):
                try:
                    result = load_flight_analytics(
                        str(DATA_PATH),
                        DATA_PATH.stat().st_mtime
                    )
                    st.session_state["flight_analytics"] = result
                except Exception as e:
                    st.error(f"Analytics loading error: {e}")

        if "flight_analytics" in st.session_state:
            (
                total_flights,
                delayed_flights,
                yearly_data,
                airline_data,
                airport_data,
                season_data
            ) = st.session_state["flight_analytics"]

            delay_rate = (
                delayed_flights / total_flights * 100
                if total_flights else 0
            )

            st.markdown(
                '<div class="section">Flight Statistics</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Total Flights", f"{total_flights:,}")

            with c2:
                st.metric("Delayed Flights", f"{delayed_flights:,}")

            with c3:
                st.metric("Delay Rate", f"{delay_rate:.2f}%")

            st.divider()

            # YEARLY ANALYSIS
            st.subheader("Yearly Flight Analysis")
            yearly_data = yearly_data.sort_values("YEAR")

            yearly_chart = yearly_data.set_index("YEAR")
            st.line_chart(yearly_chart[["count"]])
            st.bar_chart(yearly_chart[["delay_rate"]])

            st.dataframe(
                yearly_data,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # ==========================================
            # AIRLINE PERFORMANCE ANALYSIS
            # ==========================================

            st.subheader("Airline Performance")

            airline_data = airline_data.sort_values(
                "count",
                ascending=False
            ).copy()

            airline_data["delay_rate"] = (
                airline_data["sum"] / airline_data["count"] * 100
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("Flight Volume by Airline")

                st.bar_chart(
                    airline_data.set_index("OP_CARRIER")[["count"]]
                )

            with col2:
                st.write("Delay Percentage by Airline")

                airline_delay = airline_data.sort_values(
                    "delay_rate",
                    ascending=False
                )

                st.bar_chart(
                    airline_delay.set_index("OP_CARRIER")[["delay_rate"]]
                )

            st.write("Detailed Airline Performance")

            st.dataframe(
                airline_data.rename(columns={
                    "OP_CARRIER": "Airline",
                    "count": "Total Flights",
                    "sum": "Delayed Flights",
                    "delay_rate": "Delay Rate (%)"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # ==========================================
            # AIRPORT PERFORMANCE ANALYSIS
            # ==========================================

            st.subheader("Airport Performance")

            airport_data = airport_data.copy()

            airport_data["delay_rate"] = (
                airport_data["sum"] / airport_data["count"] * 100
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("Top 10 Airports by Flight Volume")

                top_volume = airport_data.sort_values(
                    "count",
                    ascending=False
                ).head(10)

                st.bar_chart(
                    top_volume.set_index("ORIGIN")[["count"]]
                )

            with col2:
                st.write("Top 10 Airports by Delay Rate")

                # Ignore airports with fewer than 1,000 flights
                reliable_airports = airport_data[
                    airport_data["count"] >= 1000
                ]

                top_delay = reliable_airports.sort_values(
                    "delay_rate",
                    ascending=False
                ).head(10)

                st.bar_chart(
                    top_delay.set_index("ORIGIN")[["delay_rate"]]
                )

            st.write("Airport Performance Details")

            st.dataframe(
                airport_data.sort_values(
                    "delay_rate",
                    ascending=False
                ).rename(columns={
                    "ORIGIN": "Airport",
                    "count": "Total Flights",
                    "sum": "Delayed Flights",
                    "delay_rate": "Delay Rate (%)"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # SEASONAL ANALYSIS
            st.subheader("Seasonal Delay Analysis")
            season_data = season_data.sort_values(
                "delay_rate",
                ascending=False
            )

            st.bar_chart(
                season_data.set_index("SEASON")[["delay_rate"]]
            )

            st.dataframe(
                season_data,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "Click 'Load Historical Analytics' to analyze "
                "your connected flight dataset."
            )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.markdown("""
    <div class="hero">
        <h1>Model Performance</h1>
        <p>Prediction engine evaluation and performance indicators.</p>
    </div>
    """, unsafe_allow_html=True)

    if not metrics:

        st.warning(
            "Model evaluation metrics are not available yet."
        )

    else:

        st.markdown(
            '<div class="section">Evaluation Results</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Accuracy", metric_value("accuracy"))

        with c2:
            st.metric("Precision", metric_value("precision"))

        with c3:
            st.metric("Recall", metric_value("recall"))

        with c4:
            st.metric("F1 Score", metric_value("f1", "f1_score"))

        st.subheader("Detailed Metrics")

        st.dataframe(
            pd.DataFrame(
                list(metrics.items()),
                columns=["Metric", "Value"]
            ),
            use_container_width=True,
            hide_index=True
        )


        st.metric("ROC-AUC", metric_value("roc_auc"))

        st.caption(
            f"Model selected: {metrics.get('model', 'Not available')}. "
            f"Training years: {metrics.get('training_years', 'Not available')}; "
            f"testing years: {metrics.get('testing_years', 'Not available')}."
        )

        results_dir = BASE / "results"

        comparison_path = results_dir / "model_comparison.json"
        if comparison_path.exists():
            try:
                with open(comparison_path, "r", encoding="utf-8") as f:
                    comparison_data = json.load(f)

                comparison_rows = []
                for model_name, values in comparison_data.items():
                    if isinstance(values, dict):
                        comparison_rows.append({
                            "Model": model_name,
                            "Accuracy": f"{values.get('accuracy', 0):.2%}",
                            "Precision": f"{values.get('precision', 0):.2%}",
                            "Recall": f"{values.get('recall', 0):.2%}",
                            "F1 Score": f"{values.get('f1', 0):.2%}",
                            "ROC-AUC": f"{values.get('roc_auc', 0):.2%}",
                            "Best CV F1": f"{values.get('best_cv_f1', 0):.2%}"
                        })

                if comparison_rows:
                    st.subheader("Model Comparison")
                    st.dataframe(
                        pd.DataFrame(comparison_rows),
                        use_container_width=True,
                        hide_index=True
                    )
            except Exception:
                st.info("Model comparison details are currently unavailable.")

        for image_name, title in [
            ("confusion_matrix.png", "Confusion Matrix"),
            ("roc_curve.png", "ROC Curve")
        ]:
            image_path = results_dir / image_name
            if image_path.exists():
                st.subheader(title)
                st.image(str(image_path), use_container_width=True)

        report_path = results_dir / "classification_report.txt"
        if report_path.exists():
            st.subheader("Classification Report")
            st.code(report_path.read_text(encoding="utf-8"))


# ==========================================
# ABOUT
# ==========================================

elif page == "About":

    st.markdown("""
    <div class="hero">
        <h1>About SKYPREDICT</h1>
        <p>Intelligent aviation analytics.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Our Platform

    SKYPREDICT is a flight intelligence application
    designed to estimate flight delay risks using
    historical aviation data and machine learning.

    ### Key Capabilities

    - Flight delay prediction
    - Airline-based analysis
    - Airport-based analysis
    - Departure-time analysis
    - Model performance monitoring

    ### Technology

    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - Machine Learning
    """)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    <b>SKYPREDICT</b><br>
    Intelligent Flight Analytics Platform<br>
    Machine-learning estimates are not live airline information.
</div>
""", unsafe_allow_html=True)