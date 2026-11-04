from pathlib import Path
import json
import joblib
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE = Path(__file__).resolve().parent

DATA = BASE / "data" / "processed" / "flight_data.csv"

MODELS = BASE / "models"
RESULTS = BASE / "results"

MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHUNK_SIZE = 500_000
SAMPLE_FRACTION = 0.002
RANDOM_STATE = 42

TRAIN_YEARS = list(range(2009, 2017))
TEST_YEARS = [2017, 2018]

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

features = [
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_HOUR",
    "DAY_OF_WEEK",
    "MONTH",
    "YEAR",
    "SEASON"
]

target = "target"

categorical = [
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "SEASON"
]

numerical = [
    "CRS_DEP_TIME",
    "DEP_HOUR",
    "DAY_OF_WEEK",
    "MONTH",
    "YEAR"
]

# --------------------------------------------------
# LOAD DATA IN CHUNKS
# --------------------------------------------------

def load_sample():

    if not DATA.exists():
        raise FileNotFoundError(
            "Dataset missing. Run python prepare_data.py"
        )

    train_samples = []
    test_samples = []

    print("\nReading dataset in chunks...")

    chunk_number = 0

    for chunk in pd.read_csv(
        DATA,
        usecols=features + [target],
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1

        chunk["YEAR"] = pd.to_numeric(
            chunk["YEAR"], errors="coerce"
        )

        chunk[target] = pd.to_numeric(
            chunk[target], errors="coerce"
        )

        chunk = chunk.dropna(
            subset=features + [target]
        )

        chunk[target] = chunk[target].astype(int)

        chunk = chunk[
            chunk[target].isin([0, 1])
        ]

        train_chunk = chunk[
            chunk["YEAR"].isin(TRAIN_YEARS)
        ]

        test_chunk = chunk[
            chunk["YEAR"].isin(TEST_YEARS)
        ]

        if not train_chunk.empty:

            sample = train_chunk.sample(
                frac=SAMPLE_FRACTION,
                random_state=RANDOM_STATE + chunk_number
            )

            train_samples.append(sample)

        if not test_chunk.empty:

            sample = test_chunk.sample(
                frac=SAMPLE_FRACTION,
                random_state=RANDOM_STATE + chunk_number
            )

            test_samples.append(sample)

        print(
            f"Processed chunk {chunk_number}"
        )

    train_df = pd.concat(
        train_samples,
        ignore_index=True
    )

    test_df = pd.concat(
        test_samples,
        ignore_index=True
    )

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Training or testing sample is empty."
        )

    print("\nSampling complete.")

    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))

    return (
        train_df[features],
        train_df[target],
        test_df[features],
        test_df[target]
    )


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def create_preprocessor():

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10
            )
        )
    ])

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    return ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            categorical
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical
        )
    ])


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities
            )
        )
    }


# --------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------

def main():

    print("=" * 60)
    print("SKYPREDICT MODEL TRAINING")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_sample()

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    # Tune on a smaller subset to control runtime
    tune_size = min(15_000, len(X_train))

    X_tune = X_train.sample(
        n=tune_size,
        random_state=RANDOM_STATE
    )

    y_tune = y_train.loc[X_tune.index]

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    model_results = {}

    best_score = -1
    best_model = None
    best_name = None

    # --------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION")
    print("=" * 50)

    lr_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),

        (
            "classifier",
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="lbfgs"
            )
        )
    ])

    lr_grid = {
        "classifier__C": [0.1, 1.0]
    }

    lr_search = GridSearchCV(
        lr_pipeline,
        lr_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    lr_search.fit(X_tune, y_tune)

    lr_best = lr_search.best_estimator_

    lr_best.fit(X_train, y_train)

    lr_metrics = evaluate_model(
        lr_best,
        X_test,
        y_test
    )

    lr_metrics["best_cv_f1"] = float(
        lr_search.best_score_
    )

    lr_metrics["best_parameters"] = {
        key: str(value)
        for key, value in lr_search.best_params_.items()
    }

    model_results["Logistic Regression"] = lr_metrics

    print("\nLogistic Regression results:")
    print(json.dumps(lr_metrics, indent=4))

    if lr_search.best_score_ > best_score:

        best_score = lr_search.best_score_
        best_model = lr_best
        best_name = "Logistic Regression"

    # --------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("RANDOM FOREST")
    print("=" * 50)

    rf_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),

        (
            "classifier",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
                n_jobs=1
            )
        )
    ])

    rf_grid = {
    "classifier__n_estimators": [40],
    "classifier__max_depth": [12, 18],
    "classifier__min_samples_leaf": [10]
}
    

    rf_search = GridSearchCV(
        rf_pipeline,
        rf_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    rf_search.fit(X_tune, y_tune)

    rf_best = rf_search.best_estimator_

    rf_best.fit(X_train, y_train)

    rf_metrics = evaluate_model(
        rf_best,
        X_test,
        y_test
    )

    rf_metrics["best_cv_f1"] = float(
        rf_search.best_score_
    )

    rf_metrics["best_parameters"] = {
        key: str(value)
        for key, value in rf_search.best_params_.items()
    }

    model_results["Random Forest"] = rf_metrics

    print("\nRandom Forest results:")
    print(json.dumps(rf_metrics, indent=4))

    if rf_search.best_score_ > best_score:

        best_score = rf_search.best_score_
        best_model = rf_best
        best_name = "Random Forest"

    # --------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------

    print("\nBest model:", best_name)

    final_metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    final_metrics.update({
        "model": best_name,
        "best_cv_f1": float(best_score),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "training_years": "2009-2016",
        "testing_years": "2017-2018"
    })

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    predictions = best_model.predict(X_test)

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)

    with open(
        RESULTS / "classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Delayed",
            "Delayed"
        ]
    )

    display.plot()

    plt.title("SKYPREDICT - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        RESULTS / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------

    RocCurveDisplay.from_estimator(
        best_model,
        X_test,
        y_test
    )

    plt.title("SKYPREDICT - ROC Curve")
    plt.tight_layout()

    plt.savefig(
        RESULTS / "roc_curve.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    classifier = best_model.named_steps["classifier"]

    preprocessor = best_model.named_steps["preprocessor"]

    if hasattr(classifier, "feature_importances_"):

        importance = classifier.feature_importances_

        names = preprocessor.get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": names,
            "Importance": importance
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        ).head(20)

        importance_df.to_csv(
            RESULTS / "feature_importance.csv",
            index=False
        )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    bundle = {
        "pipeline": best_model,

        "feature_columns": features,

        "feature_defaults": {
            "OP_CARRIER": "WN",
            "ORIGIN": "JFK",
            "DEST": "LAX",
            "CRS_DEP_TIME": 1325,
            "DEP_HOUR": 13,
            "DAY_OF_WEEK": 0,
            "MONTH": 1,
            "YEAR": 2018,
            "SEASON": "Winter"
        }
    }

    joblib.dump(
        bundle,
        MODELS / "flight_delay_model.joblib"
    )

    with open(
        MODELS / "metrics.json",
        "w"
    ) as f:

        json.dump(final_metrics, f, indent=4)

    with open(
        RESULTS / "model_comparison.json",
        "w"
    ) as f:

        json.dump(model_results, f, indent=4)

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(json.dumps(model_results, indent=4))

    print("\n" + "=" * 60)
    print("BEST MODEL:", best_name)
    print("=" * 60)

    print(json.dumps(final_metrics, indent=4))

    print("\nTraining completed successfully!")

    print("\nSaved files:")
    print("models/flight_delay_model.joblib")
    print("models/metrics.json")
    print("results/model_comparison.json")
    print("results/confusion_matrix.png")
    print("results/roc_curve.png")
    print("results/classification_report.txt")


if __name__ == "__main__":
    main()