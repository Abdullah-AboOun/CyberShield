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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from cybershield.ml.preprocessing import (
    CATEGORICAL,
    TEST_CSV,
    TRAIN_CSV,
    ZEEK_WINDOWED_FEATURES,
    build_preprocessor,
    load_csv,
    split_xy,
)

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
SEED = 42  # fixed so every run builds the same model (reproducibility, AI-03)

# Finalists picked on the validation split (notebooks/03_model_selection), then scored once on test.
FINALISTS = {
    "rf-baseline-v1": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    "hgb-v1": lambda: HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=63, max_iter=300, random_state=SEED
    ),
}

# Candidate specs: (classifier factory, feature list or None for all, decision threshold).
# hgb-zeek-v1 uses the 21 Zeek-computable features and threshold 0.68 chosen on validation (notebook 06).
MODEL_SPECS = {
    "rf-baseline-v1": (FINALISTS["rf-baseline-v1"], None, 0.50),
    "hgb-v1": (FINALISTS["hgb-v1"], None, 0.50),
    "hgb-zeek-v1": (FINALISTS["hgb-v1"], ZEEK_WINDOWED_FEATURES, 0.68),
}


def build_model(classifier=None, categorical: list[str] = CATEGORICAL) -> Pipeline:
    """Preprocessor + classifier in one object, so fit() always learns both from train data only."""
    if classifier is None:  # not `or`: unfitted forests raise on truthiness (len() needs fitted trees)
        classifier = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    return Pipeline([("preprocess", build_preprocessor(categorical=categorical)), ("classifier", classifier)])


def predict_with_threshold(model: Pipeline, X: pd.DataFrame, threshold: float = 0.50):
    """Predict binary labels using P(attack) >= threshold."""
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


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
    baseline = evaluate(y_test, dummy.predict(X_test))
    MODELS_DIR.mkdir(exist_ok=True)

    for version, (make_classifier, features, threshold) in MODEL_SPECS.items():
        cols = features if features is not None else list(X_train.columns)
        categorical = [c for c in CATEGORICAL if c in cols]
        classifier = make_classifier()
        model = build_model(classifier, categorical=categorical).fit(X_train[cols], y_train)
        pred = predict_with_threshold(model, X_test[cols], threshold)
        # Per attack family: share of each family's flows flagged as attack (for Normal, that is the FPR).
        per_family = (pd.Series(pred, index=test.index) == 1).groupby(test["attack_cat"]).mean()
        report = {
            "model_version": version,
            "features": cols,
            "n_features": len(cols),
            "threshold": threshold,
            "params": classifier.get_params(),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "baseline_most_frequent": baseline,
            "test": evaluate(y_test, pred),
            "flagged_as_attack_by_family": per_family.round(4).to_dict(),
        }
        joblib.dump(model, MODELS_DIR / f"{version}.joblib")
        (MODELS_DIR / f"{version}.metrics.json").write_text(json.dumps(report, indent=2, default=str))
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("model_version", "n_features", "threshold", "test", "flagged_as_attack_by_family")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
