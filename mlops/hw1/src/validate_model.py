import json
import os
import pickle
import sys

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

METRICS_PATH = "metrics/metrics.json"


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_accuracy(model, df, params):
    X = df[["total_bill", "size"]]
    y = df["high_tip"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def validate_model():
    params = load_params()
    accuracy_min = params.get("accuracy_min")
    if accuracy_min is None:
        raise ValueError("accuracy_min is missing in params.yaml")

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/processed/dataset.csv")

    metrics = load_metrics()
    accuracy = None
    if isinstance(metrics, dict):
        accuracy = metrics.get("accuracy")

    if accuracy is None:
        accuracy = compute_accuracy(model, df, params)
    else:
        accuracy = float(accuracy)

    if accuracy < accuracy_min:
        print(
            "Model validation failed: "
            f"accuracy {accuracy:.4f} < accuracy_min {accuracy_min:.4f}"
        )
        sys.exit(1)

    print(
        "Model validation passed: "
        f"accuracy {accuracy:.4f} >= accuracy_min {accuracy_min:.4f}"
    )


if __name__ == "__main__":
    validate_model()
