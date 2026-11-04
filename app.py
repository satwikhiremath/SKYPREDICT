import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from common import load_model_bundle, build_prediction_input, predict_flight


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SKYPREDICT | Flight Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "flight_delay_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

ANALYTICS_DIR = BASE_DIR / "data" / "analytics"

OVERVIEW_PATH = ANALYTICS_DIR / "overview_stats.json"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.metric-card {
    background-color: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.metric-label {
    font-size: 14px;
    color: #64748b;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 8px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
}

.hero-description {
    color: #64748b;
    font-size: 16px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA LOADING FUNCTIONS
# =========================================================

@st.cache_data
def load_json_file(file_path):

    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return {}


@st.cache_data
def load_csv_file(file_path):

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)

    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_prediction_model():

    if not MODEL_PATH.exists():
        return None

    try:
        return load_model_bundle(MODEL_PATH)

    except Exception:
        return None


# =========================================================
# LOAD PROJECT DATA
# =========================================================

overview = load_json_file(OVERVIEW_PATH)
metrics = load_json_file(METRICS_PATH)

model_bundle = load_prediction_model()

airline_data = load_csv_file(
    ANALYTICS_DIR / "airline_stats.csv"
)

origin_data = load_csv_file(
    ANALYTICS_DIR / "origin_airport_stats.csv"
)

destination_data = load_csv_file(
    ANALYTICS_DIR / "destination_airport_stats.csv"
)

monthly_data = load_csv_file(
    ANALYTICS_DIR / "monthly_stats.csv"
)

yearly_data = load_csv_file(
    ANALYTICS_DIR / "yearly_stats.csv"
)

route_data = load_csv_file(
    ANALYTICS_DIR / "top_route_stats.csv"
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_number(value):

    try:
        return f"{int(float(value)):,}"

    except (ValueError, TypeError):
        return "0"


def show_metric(title, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_chart(data, x_column, y_column, title, chart_type="bar"):

    if data.empty:
        st.info("Analytics data is not available.")
        return

    if x_column not in data.columns or y_column not in data.columns:
        st.info("Required chart columns are not available.")
        return

    if chart_type == "line":
        fig = px.line(
            data,
            x=x_column,
            y=y_column,
            markers=True,
            title=title
        )

    else:
        fig = px.bar(
            data,
            x=x_column,
            y=y_column,
            title=title
        )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def show_first_available_chart(data, title):

    if data.empty:
        st.info(f"{title}: Data not available.")
        return

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = data.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if not numeric_columns or not categorical_columns:
        st.info(f"{title}: Suitable chart columns not found.")
        return

    x_column = categorical_columns[0]
    y_column = numeric_columns[0]

    chart_data = data.head(15)

    show_chart(
        chart_data,
        x_column,
        y_column,
        title
    )


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("# ✈️ SKYPREDICT")

st.sidebar.caption(
    "Intelligent Flight Analytics Platform"
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Predict Flight",
        "Flight Insights",
        "Model Performance",
        "About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "SKYPREDICT | Flight Intelligence"
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        """
        <div style="font-size:11px;letter-spacing:2px;color:#64748b;">
            FLIGHT INTELLIGENCE PLATFORM
        </div>

        <div class="hero-title">
            Know Before You Fly.
        </div>

        <div class="hero-description">
            Intelligent flight analytics and machine-learning-based
            flight delay predictions.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Your Dashboard")

    # Read actual values from overview_stats.json

    total_flights = overview.get("total_flights", 0)

    delayed_flights = overview.get("delayed_flights", 0)

    on_time_flights = overview.get(
        "not_delayed_flights", 0
    )

    delay_percentage = overview.get(
        "delay_rate_pct", 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric(
            "Total Flights",
            format_number(total_flights)
        )

    with col2:
        show_metric(
            "Delayed Flights",
            format_number(delayed_flights)
        )

    with col3:
        show_metric(
            "On-Time Flights",
            format_number(on_time_flights)
        )

    with col4:
        show_metric(
            "Delay Percentage",
            f"{float(delay_percentage):.2f}%"
        )

    if not overview:
        st.warning(
            "Dashboard statistics could not be loaded. "
            "Check data/analytics/overview_stats.json."
        )

    st.markdown("---")

    st.subheader("Historical Flight Trends")

    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:

        if not monthly_data.empty:

            show_first_available_chart(
                monthly_data,
                "Monthly Flight Statistics"
            )

        else:
            st.info("Monthly statistics are unavailable.")

    with trend_col2:

        if not yearly_data.empty:

            show_first_available_chart(
                yearly_data,
                "Yearly Flight Statistics"
            )

        else:
            st.info("Yearly statistics are unavailable.")

    st.markdown("---")

    st.subheader("Prediction Model Snapshot")

    c1, c2, c3 = st.columns(3)

    with c1:
        show_metric(
            "Accuracy",
            f"{metrics.get('accuracy', 0) * 100:.2f}%"
        )

    with c2:
        show_metric(
            "Precision",
            f"{metrics.get('precision', 0) * 100:.2f}%"
        )

    with c3:
        show_metric(
            "Recall",
            f"{metrics.get('recall', 0) * 100:.2f}%"
        )


# =========================================================
# PREDICT FLIGHT
# =========================================================

elif page == "Predict Flight":

    st.title("✈️ Flight Delay Prediction")

    st.write(
        "Enter flight details to estimate the possibility "
        "of a departure delay."
    )

    if model_bundle is None:

        st.error(
            "Prediction model not found. "
            "Check models/flight_delay_model.joblib."
        )

    else:

        carrier_options = [
            "AA", "DL", "UA", "WN", "AS",
            "B6", "NK", "F9", "G4", "HA",
            "OO", "YX", "9E", "EV"
        ]

        col1, col2 = st.columns(2)

        with col1:

            carrier = st.selectbox(
                "Airline Carrier",
                carrier_options
            )

            origin = st.text_input(
                "Origin Airport Code",
                value="ATL",
                max_chars=3
            ).upper().strip()

        with col2:

            destination = st.text_input(
                "Destination Airport Code",
                value="ORD",
                max_chars=3
            ).upper().strip()

            departure_time = st.time_input(
                "Scheduled Departure Time"
            )

        if st.button(
            "Predict Flight Delay",
            type="primary",
            use_container_width=True
        ):

            if (
                len(origin) != 3
                or len(destination) != 3
            ):

                st.error(
                    "Enter valid three-letter airport codes."
                )

            elif origin == destination:

                st.error(
                    "Origin and destination cannot be the same."
                )

            else:

                try:

                    departure_hhmm = (
                        departure_time.hour * 100
                        + departure_time.minute
                    )

                    features = build_prediction_input(
                        model_bundle,
                        carrier,
                        origin,
                        destination,
                        departure_hhmm
                    )

                    prediction, probability = predict_flight(
                        model_bundle,
                        features
                    )

                    st.markdown("---")

                    if int(prediction) == 1:

                        st.error(
                            "⚠️ Flight Delay Likely"
                        )

                    else:

                        st.success(
                            "✅ Flight Likely to Be On Time"
                        )

                    if probability is not None:

                        st.metric(
                            "Estimated Delay Probability",
                            f"{probability * 100:.2f}%"
                        )

                        st.progress(
                            min(max(probability, 0.0), 1.0)
                        )

                    st.caption(
                        "This prediction is generated by a "
                        "machine-learning model using historical data."
                    )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )


# =========================================================
# FLIGHT INSIGHTS
# =========================================================

elif page == "Flight Insights":

    st.title("📊 Flight Insights")

    st.write(
        "Explore historical flight patterns and statistics."
    )

    st.markdown("---")

    st.subheader("Airline Statistics")

    show_first_available_chart(
        airline_data,
        "Flight Statistics by Airline"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Origin Airport Statistics")

        show_first_available_chart(
            origin_data,
            "Top Origin Airports"
        )

    with col2:

        st.subheader("Destination Airport Statistics")

        show_first_available_chart(
            destination_data,
            "Top Destination Airports"
        )

    st.markdown("---")

    st.subheader("Monthly Flight Statistics")

    show_first_available_chart(
        monthly_data,
        "Monthly Flight Trends"
    )

    st.markdown("---")

    st.subheader("Yearly Flight Statistics")

    show_first_available_chart(
        yearly_data,
        "Yearly Flight Trends"
    )

    st.markdown("---")

    st.subheader("Popular Flight Routes")

    if not route_data.empty:

        st.dataframe(
            route_data,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Route statistics are not available."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Evaluation results of the trained flight delay "
        "classification model."
    )

    if not metrics:

        st.warning(
            "Model evaluation metrics are unavailable."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            show_metric(
                "Accuracy",
                f"{metrics.get('accuracy', 0) * 100:.2f}%"
            )

        with col2:

            show_metric(
                "Precision",
                f"{metrics.get('precision', 0) * 100:.2f}%"
            )

        with col3:

            show_metric(
                "Recall",
                f"{metrics.get('recall', 0) * 100:.2f}%"
            )

        col4, col5, col6 = st.columns(3)

        with col4:

            show_metric(
                "F1 Score",
                f"{metrics.get('f1', 0) * 100:.2f}%"
            )

        with col5:

            show_metric(
                "ROC-AUC",
                f"{metrics.get('roc_auc', 0) * 100:.2f}%"
            )

        with col6:

            show_metric(
                "Best CV F1",
                f"{metrics.get('best_cv_f1', 0) * 100:.2f}%"
            )

        st.markdown("---")

        st.subheader("Training Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:

            st.write(
                "**Model:**",
                metrics.get("model", "Not available")
            )

            st.write(
                "**Training Rows:**",
                format_number(
                    metrics.get("training_rows", 0)
                )
            )

            st.write(
                "**Testing Rows:**",
                format_number(
                    metrics.get("testing_rows", 0)
                )
            )

        with info_col2:

            st.write(
                "**Training Years:**",
                metrics.get("training_years", "Not available")
            )

            st.write(
                "**Testing Years:**",
                metrics.get("testing_years", "Not available")
            )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.title("About SKYPREDICT")

    st.markdown("""
    **SKYPREDICT** is an intelligent flight analytics
    and prediction platform designed to estimate the
    possibility of flight departure delays.

    ### Project Objectives

    - Analyze historical airline flight data.
    - Identify flight delay patterns.
    - Predict potential departure delays.
    - Provide meaningful flight analytics.
    - Support better travel planning through data.

    ### Technology Stack

    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Plotly
    - Machine Learning

    ### Machine Learning

    The application uses a trained Logistic Regression
    classification pipeline to estimate whether a flight
    is likely to experience a departure delay.

    ### Project Team

    - Satwik Hiremath
    - Nitish Yadav

    ### Project

    Flight Delay Prediction and Disruption Risk
    Classification System.
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        <b>SKYPREDICT</b> · Intelligent Flight Analytics
        <br><br>
        Machine-learning-based estimates.
        Not live airline information.
    </div>
    """,
    unsafe_allow_html=True
)
