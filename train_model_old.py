from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "flight_data.csv"
MODELS = BASE / "models"

MODELS.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(
        "Processed data missing. Run python prepare_data.py first."
    )

print("Loading prepared flight dataset...")
df = pd.read_csv(DATA)

features = [
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME"
]

target = "target"

df = df.dropna(subset=features + [target])

if df[target].nunique() != 2:
    raise ValueError("Training data must contain both delay classes 0 and 1.")

X = df[features]
y = df[target].astype(int)

print(f"Dataset records: {len(df):,}")
print("\nSplitting training and testing data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

categorical = ["OP_CARRIER", "ORIGIN", "DEST"]
numerical = ["CRS_DEP_TIME"]

preprocessor = ColumnTransformer([
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    ),
    (
        "numerical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numerical
    )
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

print("\nTraining Logistic Regression model...")
pipeline.fit(X_train, y_train)

print("Evaluating model...")
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(
        y_test, predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, predictions, zero_division=0
    )),
    "f1": float(f1_score(
        y_test, predictions, zero_division=0
    )),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "model": "Logistic Regression"
}

bundle = {
    "pipeline": pipeline,
    "feature_columns": features,
    "feature_defaults": {
        "OP_CARRIER": "WN",
        "ORIGIN": "JFK",
        "DEST": "LAX",
        "CRS_DEP_TIME": 1325
    }
}

joblib.dump(
    bundle,
    MODELS / "flight_delay_model.joblib"
)

with open(MODELS / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nTRAINING COMPLETE")
print(json.dumps(metrics, indent=4))
print("\nModel saved successfully!")
