import joblib
import pandas as pd
from datetime import date


def load_model_bundle(path):
    return joblib.load(path)


def build_prediction_input(
    bundle,
    carrier,
    origin,
    destination,
    departure_hhmm,
    flight_date=None
):
    if flight_date is None:
        flight_date = date(2018, 1, 1)

    if isinstance(flight_date, str):
        flight_date = date.fromisoformat(flight_date)

    departure_hour = departure_hhmm // 100

    day_of_week = flight_date.weekday()
    month = flight_date.month
    year = flight_date.year

    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Autumn"

    row = {
        "OP_CARRIER": carrier,
        "ORIGIN": origin,
        "DEST": destination,
        "CRS_DEP_TIME": departure_hhmm,
        "DEP_HOUR": departure_hour,
        "DAY_OF_WEEK": day_of_week,
        "MONTH": month,
        "YEAR": year,
        "SEASON": season
    }

    columns = bundle["feature_columns"]

    return pd.DataFrame([row], columns=columns)


def predict_flight(bundle, X):
    return bundle["pipeline"].predict(X)


def predict_flight_probability(bundle, X):
    probabilities = bundle["pipeline"].predict_proba(X)[0]
    classes = list(bundle["pipeline"].classes_)

    if 1 in classes:
        return probabilities[classes.index(1)]

    return 0.0