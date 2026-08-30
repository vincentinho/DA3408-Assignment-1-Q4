import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import subprocess

# -------------------------
# Reproducibility settings
# -------------------------

SEED = 42

HIDDEN_UNUSED = None  # intentionally nothing complicated


# -------------------------
# MLflow
# -------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("reproducibility-capstone")


# -------------------------
# Load data
# -------------------------

df = pd.read_csv("data/iris.csv")

X = df.drop(columns=["target"])
y = df["target"]


# -------------------------
# Train/test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)


# -------------------------
# Git commit
# -------------------------

def get_git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip()


# -------------------------
# Model
# -------------------------

model = LogisticRegression(
    max_iter=200,
    random_state=SEED,
)


# -------------------------
# Training + MLflow
# -------------------------

with mlflow.start_run() as run:

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("seed", SEED)

    mlflow.set_tag("git_commit", get_git_commit())

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="IrisReproducibilityModel"
    )

    print("Run ID:", run.info.run_id)
    print("Accuracy:", accuracy)
