"""Train, evaluate, and score the 19-feature Native Zeek Hybrid model (zeek-hybrid-m3-v1).

Run:  uv run python -m cybershield.ml.train_zeek
1. CIC-IDS2017 leave-one-day-out: each attack day is scored by a model that never saw that day, so its
   attack types are new to it. (A random split leaks: neighbouring minutes of one attack land on both sides.)
2. Final model: all 5 CIC days + balanced UWF-ZeekData24 & UWF-ZeekDataFall24-2, scored on unseen
   UWF-Sum25-1/2 and AIT-LDSv2 (recall-only). Artefacts go to models/.
Every test reports per-flow, per-host-minute, and per-attack-type recall (floods dominate per-flow numbers).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from cybershield.ml.zeek_features import (
    ZEEK_CAT_COLS,
    ZEEK_HYBRID_FEATURES,
    add_zeek_hybrid_features,
    load_ait_zeek,
    load_cic2017_zeek,
    load_uwf_zeek,
    sample_balanced_uwf,
)

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_VERSION = "zeek-hybrid-m3-v1"
SEED = 42
CIC_ATTACK_DAYS = ["cic_tuesday", "cic_wednesday", "cic_thursday", "cic_friday"]


def build_zeek_pipeline(features: list[str] = ZEEK_HYBRID_FEATURES, max_iter: int = 200, random_state: int = SEED) -> Pipeline:
    """Build an unfitted pipeline: one-hot the categorical Zeek columns in `features`, pass numbers through."""
    cat = [c for c in ZEEK_CAT_COLS if c in features]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", "passthrough", [c for c in features if c not in cat]),
        ]
    )
    clf = HistGradientBoostingClassifier(random_state=random_state, max_iter=max_iter)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_zeek_model(train: pd.DataFrame, features: list[str] = ZEEK_HYBRID_FEATURES) -> Pipeline:
    return build_zeek_pipeline(features).fit(train[features], train["label"].to_numpy(dtype=int))


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.50) -> dict:
    """Compute ROC-AUC, F1, Precision, Recall, FPR, and confusion matrix."""
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4) if len(np.unique(y_true)) > 1 else None,
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "false_positive_rate": round(fp / (fp + tn), 5) if (fp + tn) else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def evaluate_host_minutes(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.50) -> dict:
    """Score per (source, src_ip, minute): attack if any flow is, flagged if any flow is.

    This matches one alert per host-minute and stops a 900k-flow scan counting 900k times.
    """
    keys = [df["source"].to_numpy(), df["src_ip"].to_numpy(), df["window_id"].to_numpy()]
    g = pd.DataFrame({"y": df["label"].to_numpy(dtype=int), "p": probs}).groupby(keys).max()
    return evaluate_predictions(g["y"].to_numpy(), g["p"].to_numpy(), threshold)


def evaluate_test_set(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.50) -> dict:
    """Per-flow metrics, per-host-minute metrics, recall per attack type, and their unweighted mean."""
    is_attack = df["label"].to_numpy() == 1
    flagged = probs[is_attack] >= threshold
    by_type = pd.Series(flagged).groupby(df["attack_type"].to_numpy(dtype=str)[is_attack]).mean().round(4).to_dict()
    return {
        "flow": evaluate_predictions(df["label"].to_numpy(dtype=int), probs, threshold),
        "host_minute": evaluate_host_minutes(df, probs, threshold),
        "recall_by_attack": by_type,
        "macro_recall": round(float(np.mean(list(by_type.values()))), 4) if by_type else None,
    }


def load_uwf_train(seed: int = SEED) -> pd.DataFrame:
    """Balanced 80k/80k chunks of UWF-ZeekData24 and UWF-ZeekDataFall24-2 used as co-training data."""
    parts = [sample_balanced_uwf(load_uwf_zeek(n), max_per_class=80_000, seed=seed) for n in ("UWF-ZeekData24", "UWF-ZeekDataFall24-2")]
    return pd.concat(parts, ignore_index=True)


def run_lodo(cic: pd.DataFrame, extra_train: pd.DataFrame, features: list[str] = ZEEK_HYBRID_FEATURES):
    """Leave-one-day-out on CIC attack days. Returns ({day: report}, {day: fitted model})."""
    reports, models = {}, {}
    for day in CIC_ATTACK_DAYS:
        test = cic[cic["source"] == day]
        models[day] = fit_zeek_model(pd.concat([cic[cic["source"] != day], extra_train], ignore_index=True), features)
        reports[day] = evaluate_test_set(test, models[day].predict_proba(test[features])[:, 1])
    return reports, models


def score_zeek_flows(
    model: Pipeline,
    raw_conn_df: pd.DataFrame,
    high_threshold: float = 0.90,
    suspicious_threshold: float = 0.60,
    allowlist_src_ips: set[str] | None = None,
) -> pd.DataFrame:
    """Score raw Zeek conn.log rows and assign alert severity tiers ('normal', 'suspicious', 'high').

    Optionally skips alerting for known high-volume gateway/router IPs in `allowlist_src_ips`.
    """
    # shortcut: window stats only see this batch, so pass whole minutes; buffer >=60s once scoring live.
    featured = add_zeek_hybrid_features(raw_conn_df, group_col=None)
    probs = model.predict_proba(featured[ZEEK_HYBRID_FEATURES])[:, 1]
    severity = np.where(
        probs >= high_threshold,
        "high",
        np.where(probs >= suspicious_threshold, "suspicious", "normal"),
    )
    if allowlist_src_ips:
        is_allowed = featured["src_ip"].isin(allowlist_src_ips).to_numpy()
        severity = np.where(is_allowed, "normal", severity)

    out = featured.copy()
    out["attack_prob"] = probs.astype(np.float32)
    out["severity"] = severity
    return out


def main() -> None:
    cic = load_cic2017_zeek()
    uwf_train = load_uwf_train()

    lodo, _ = run_lodo(cic, uwf_train)
    print(json.dumps({"cic_leave_one_day_out": lodo}, indent=2), flush=True)

    train_df = pd.concat([cic, uwf_train], ignore_index=True)
    model = fit_zeek_model(train_df)

    test_sets = {
        "uwf_sum25_1": load_uwf_zeek("UWF-ZeekDataSum25-1"),
        "uwf_sum25_2": load_uwf_zeek("UWF-ZeekDataSum25-2"),
        "ait_russellmitchell_recall_only": load_ait_zeek(),
    }
    evals = {
        name: evaluate_test_set(tdf, model.predict_proba(tdf[ZEEK_HYBRID_FEATURES])[:, 1])
        for name, tdf in test_sets.items()
    }

    MODELS_DIR.mkdir(exist_ok=True)
    report = {
        "model_version": MODEL_VERSION,
        "features": ZEEK_HYBRID_FEATURES,
        "n_features": len(ZEEK_HYBRID_FEATURES),
        "train_rows": len(train_df),
        "threshold": 0.50,
        "recommended_thresholds_unvalidated": {"suspicious": 0.60, "high": 0.90},
        "cic_leave_one_day_out": lodo,
        "evaluations": evals,
    }
    joblib.dump(model, MODELS_DIR / f"{MODEL_VERSION}.joblib")
    (MODELS_DIR / f"{MODEL_VERSION}.metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"evaluations": evals}, indent=2))


if __name__ == "__main__":
    main()
