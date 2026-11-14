import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def get_project_path():

    return os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )


def load_data():

    file_path = os.path.join(
        get_project_path(),
        "data",
        "processed_transactions.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Run preprocessing.py first."
        )

    return pd.read_csv(file_path)


def prepare_features(df):

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    # Time-based features
    data["transaction_hour"] = (
        data["timestamp"].dt.hour
    )

    data["transaction_day"] = (
        data["timestamp"].dt.dayofweek
    )

    # Cross-border indicator
    data["cross_border"] = (
        data["sender_country"] !=
        data["receiver_country"]
    ).astype(int)

    # Account activity features
    account_activity = pd.concat([
        data["sender_account"],
        data["receiver_account"]
    ]).value_counts()

    data["sender_activity"] = (
        data["sender_account"].map(account_activity)
    )

    data["receiver_activity"] = (
        data["receiver_account"].map(account_activity)
    )

    # Features used for training
    feature_columns = [
        "amount",
        "transaction_hour",
        "transaction_day",
        "cross_border",
        "sender_activity",
        "receiver_activity",
        "transaction_type",
        "sender_country",
        "receiver_country"
    ]

    X = data[feature_columns].copy()

    y = data["is_suspicious"].astype(int)

    return X, y


def train_model(X, y):

    numeric_features = [
        "amount",
        "transaction_hour",
        "transaction_day",
        "cross_border",
        "sender_activity",
        "receiver_activity"
    ]

    categorical_features = [
        "transaction_type",
        "sender_country",
        "receiver_country"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    return pipeline, X_test, y_test, predictions


def evaluate_model(model, X_test, y_test, predictions):

    print("\n========== MODEL EVALUATION ==========")

    print(
        "\nAccuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print(
        "F1 Score:",
        round(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


def save_model(model):

    model_path = os.path.join(
        get_project_path(),
        "models",
        "aml_random_forest.pkl"
    )

    joblib.dump(model, model_path)

    print("\nModel Saved Successfully:")
    print(model_path)


def save_predictions(model, X):

    results = X.copy()

    results["predicted_suspicious"] = (
        model.predict(X)
    )

    results["suspicious_probability"] = (
        model.predict_proba(X)[:, 1]
    )

    output_path = os.path.join(
        get_project_path(),
        "data",
        "ml_predictions.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nPredictions Saved:")
    print(output_path)


if __name__ == "__main__":

    df = load_data()

    X, y = prepare_features(df)

    model, X_test, y_test, predictions = train_model(
        X,
        y
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        predictions
    )

    save_model(model)

    save_predictions(model, X)

    print("\nMachine Learning Training Completed!")