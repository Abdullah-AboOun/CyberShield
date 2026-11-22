"""Train, evaluate, and score the 19-feature Native Zeek Hybrid model (zeek-hybrid-m3-v1).

Run:  uv run python -m cybershield.ml.train_zeek
Trains on CIC-IDS2017 (5-day native Zeek) + balanced UWF-ZeekData24 & UWF-ZeekDataFall24-2,
evaluates on held-out CIC-IDS2017 and unseen 2025 UWF-Sum25-1/2, and saves artefacts under models/.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from cybershield.ml.zeek_features import (
    ZEEK_CAT_COLS,
    ZEEK_HYBRID_FEATURES,
    ZEEK_NUM_FEATURES,
    add_zeek_hybrid_features,
    load_cic2017_zeek,
    load_uwf_zeek,
    sample_balanced_uwf,
)

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_VERSION = "zeek-hybrid-m3-v1"
SEED = 42


def build_zeek_pipeline(max_iter: int = 200, random_state: int = SEED) -> Pipeline:
    """Build an unfitted 19-feature Native Zeek Hybrid pipeline."""
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ZEEK_CAT_COLS),
            ("num", "passthrough", ZEEK_NUM_FEATURES),
        ]
    )
    clf = HistGradientBoostingClassifier(random_state=random_state, max_iter=max_iter)
    return Pipeline([("pre", pre), ("clf", clf)])


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
    groups = cic["source"].astype(str) + "_" + cic["src_ip"].astype(str) + "_" + cic["window_id"].astype(str)
    tr_idx, te_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(cic, cic["label"], groups))

    train_df = pd.concat(
        [
            cic.iloc[tr_idx],
            sample_balanced_uwf(load_uwf_zeek("UWF-ZeekData24"), max_per_class=80_000, seed=SEED),
            sample_balanced_uwf(load_uwf_zeek("UWF-ZeekDataFall24-2"), max_per_class=80_000, seed=SEED),
        ],
        ignore_index=True,
    )

    model = build_zeek_pipeline()
    model.fit(train_df[ZEEK_HYBRID_FEATURES], train_df["label"].to_numpy(dtype=int))

    test_sets = {
        "cic2017_heldout": cic.iloc[te_idx],
        "uwf_sum25_1": load_uwf_zeek("UWF-ZeekDataSum25-1"),
        "uwf_sum25_2": load_uwf_zeek("UWF-ZeekDataSum25-2"),
    }
    eval_report = {}
    for name, tdf in test_sets.items():
        probs = model.predict_proba(tdf[ZEEK_HYBRID_FEATURES])[:, 1]
        eval_report[name] = evaluate_predictions(tdf["label"].to_numpy(dtype=int), probs, threshold=0.50)

    MODELS_DIR.mkdir(exist_ok=True)
    report = {
        "model_version": MODEL_VERSION,
        "features": ZEEK_HYBRID_FEATURES,
        "n_features": len(ZEEK_HYBRID_FEATURES),
        "train_rows": len(train_df),
        "recommended_thresholds": {"suspicious": 0.60, "high": 0.90},
        "evaluations": eval_report,
    }
    joblib.dump(model, MODELS_DIR / f"{MODEL_VERSION}.joblib")
    (MODELS_DIR / f"{MODEL_VERSION}.metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
