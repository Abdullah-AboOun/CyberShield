import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from cybershield.ml.train import build_model, evaluate


def test_evaluate_counts_and_rates():
    m = evaluate(pd.Series([0, 0, 0, 1, 1]), [0, 1, 0, 1, 0])
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert abs(m["false_positive_rate"] - 1 / 3) < 1e-9


def test_pipeline_learns_an_obvious_pattern():
    # attacks send many bytes, normal flows few: any working model must get this right
    X = pd.DataFrame({
        "proto": ["tcp"] * 20, "service": ["-"] * 20, "state": ["FIN"] * 20,
        "sbytes": [10] * 10 + [9000] * 10,
    })
    y = pd.Series([0] * 10 + [1] * 10)
    assert (build_model().fit(X, y).predict(X) == y).all()
    # passing an unfitted forest must work too (regression: `classifier or ...` crashed)
    assert (build_model(RandomForestClassifier(n_estimators=5)).fit(X, y).predict(X) == y).all()
