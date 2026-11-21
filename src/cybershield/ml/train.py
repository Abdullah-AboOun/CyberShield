"""Train and evaluate the baseline Random Forest (tasks AI-03, AI-04).

Run:  uv run python -m cybershield.ml.train
Fits on the official training split only, evaluates on the official test split,
saves the model and its metrics under models/.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from cybershield.ml.preprocessing import TEST_CSV, TRAIN_CSV, build_preprocessor, load_csv, split_xy

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_VERSION = "rf-baseline-v1"
SEED = 42  # fixed so every run builds the same forest (reproducibility, AI-03)


def build_model(classifier=None) -> Pipeline:
    """Preprocessor + classifier in one object, so fit() always learns both from train data only."""
    classifier = classifier or RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    return Pipeline([("preprocess", build_preprocessor()), ("classifier", classifier)])


def evaluate(y_true: pd.Series, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def main() -> None:
    train, test = load_csv(TRAIN_CSV), load_csv(TEST_CSV)
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    # Baseline: always predicts the most common training class. A real model must beat this.
    dummy = build_model(DummyClassifier(strategy="most_frequent")).fit(X_train, y_train)
    model = build_model().fit(X_train, y_train)
    pred = model.predict(X_test)

    # Per attack family: share of each family's flows flagged as attack (for Normal, that is the FPR).
    per_family = (pd.Series(pred, index=test.index) == 1).groupby(test["attack_cat"]).mean()

    report = {
        "model_version": MODEL_VERSION,
        "params": {"n_estimators": 100, "random_state": SEED},
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "baseline_most_frequent": evaluate(y_test, dummy.predict(X_test)),
        "random_forest": evaluate(y_test, pred),
        "flagged_as_attack_by_family": per_family.round(4).to_dict(),
    }
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{MODEL_VERSION}.joblib")
    (MODELS_DIR / f"{MODEL_VERSION}.metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
