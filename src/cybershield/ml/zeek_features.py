"""Native Zeek conn.log feature engineering and dataset loaders (19-feature Hybrid M3).

Single source of truth shared by training (`train_zeek.py`), live inference, and notebooks.
"""

from pathlib import Path
import glob

import numpy as np
import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"

ZEEK_CAT_COLS = ["proto", "service", "conn_state"]
ZEEK_RAW_NUM_COLS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]
ZEEK_FLOW_RATIO_COLS = ["orig_bytes_per_pkt", "resp_bytes_per_pkt", "byte_ratio"]
ZEEK_HOST_CTX_COLS = [
    "host_conn_60s",
    "host_uniq_ports_60s",
    "host_failed_ratio_60s",
    "host_orig_bytes_60s",
    "host_resp_bytes_60s",
]

# 19 clean Zeek Hybrid features (drops host_uniq_ips_60s & host_ports_per_ip_60s lab shortcuts).
ZEEK_HYBRID_FEATURES = ZEEK_CAT_COLS + ZEEK_RAW_NUM_COLS + ZEEK_FLOW_RATIO_COLS + ZEEK_HOST_CTX_COLS
ZEEK_NUM_FEATURES = ZEEK_RAW_NUM_COLS + ZEEK_FLOW_RATIO_COLS + ZEEK_HOST_CTX_COLS

FAILED_CONN_STATES = frozenset({"S0", "REJ", "RSTO", "RSTR", "SH", "SHR", "RSTOS0", "RSTRH"})
DEFAULT_WINDOW_SEC = 60

# Maps raw Zeek JSON / UWF / TON_IoT column variants onto our canonical internal names.
_COLUMN_ALIASES = {
    "id.orig_h": "src_ip",
    "id.orig_p": "src_port",
    "id.resp_h": "dst_ip",
    "id.resp_p": "dst_port",
    "src_ip_zeek": "src_ip",
    "dest_ip_zeek": "dst_ip",
    "dest_port_zeek": "dst_port",
    "src_bytes": "orig_bytes",
    "dst_bytes": "resp_bytes",
    "src_pkts": "orig_pkts",
    "src_ip_bytes": "orig_ip_bytes",
    "dst_pkts": "resp_pkts",
    "dst_ip_bytes": "resp_ip_bytes",
}


def add_zeek_hybrid_features(
    df: pd.DataFrame,
    window_sec: int = DEFAULT_WINDOW_SEC,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Normalize Zeek conn.log fields and attach 60-second per-host behavioral context.

    Works on both training datasets (pass `group_col="source"`) and live Zeek `conn.log`
    DataFrames (`group_col=None`). Never includes raw IPs, ports, or timestamps in the
    resulting `ZEEK_HYBRID_FEATURES`.
    """
    out = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns and v not in df.columns}).copy()

    for col in ZEEK_CAT_COLS:
        if col not in out.columns:
            out[col] = "-"
        out[col] = out[col].fillna("-").astype(str).replace({"": "-", "nan": "-", "None": "-"}).astype(object)

    for col in ZEEK_RAW_NUM_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col].replace("-", np.nan), errors="coerce").fillna(0.0).astype(np.float32)

    out["orig_bytes_per_pkt"] = (out["orig_bytes"] / out["orig_pkts"].clip(lower=1.0)).astype(np.float32)
    out["resp_bytes_per_pkt"] = (out["resp_bytes"] / out["resp_pkts"].clip(lower=1.0)).astype(np.float32)
    out["byte_ratio"] = (out["orig_bytes"] / (out["orig_bytes"] + out["resp_bytes"] + 1.0)).astype(np.float32)

    ts_num = pd.to_numeric(out["ts"], errors="coerce").fillna(0.0)
    out["window_id"] = (ts_num // window_sec).astype("int64")
    is_failed = out["conn_state"].isin(FAILED_CONN_STATES).astype(np.float32)
    out["_is_failed"] = is_failed

    keys = ([group_col] if group_col else []) + ["src_ip", "window_id"]
    host_ctx = (
        out.groupby(keys, sort=False)
        .agg(
            host_conn_60s=("ts", "size"),
            host_uniq_ports_60s=("dst_port", "nunique"),
            host_failed_ratio_60s=("_is_failed", "mean"),
            host_orig_bytes_60s=("orig_bytes", "sum"),
            host_resp_bytes_60s=("resp_bytes", "sum"),
        )
        .reset_index()
    )
    for col in ZEEK_HOST_CTX_COLS:
        host_ctx[col] = host_ctx[col].astype(np.float32)

    out = out.drop(columns=["_is_failed"]).merge(host_ctx, on=keys, how="left")
    return out


def load_cic2017_zeek(data_dir: Path | None = None, with_features: bool = True) -> pd.DataFrame:
    """Load the 5-day native Zeek CIC-IDS2017 Parquet files (labeled from UNB official GT)."""
    base = data_dir or (DATASETS_DIR / "cic2017_zeek")
    days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    df = pd.concat([pd.read_parquet(base / f"cic2017_{d}_zeek.parquet") for d in days], ignore_index=True)
    df["source"] = "cic_" + df["day"].astype(str)
    return add_zeek_hybrid_features(df, group_col="source") if with_features else df


def load_uwf_zeek(dataset_name: str, data_dir: Path | None = None, with_features: bool = True) -> pd.DataFrame:
    """Load a UWF-ZeekData Parquet capture and drop multi-tactic Duplicate rows."""
    base = data_dir or (DATASETS_DIR / "uwf_zeek")
    files = sorted(glob.glob(str(base / dataset_name / "parquet" / "*" / "*.parquet")))
    cols = [
        "ts", "src_ip_zeek", "dest_ip_zeek", "dest_port_zeek",
        "proto", "service", "conn_state", "duration", "orig_bytes", "resp_bytes",
        "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
        "label_binary", "label_tactic",
    ]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df = df[df["label_binary"] != "Duplicate"].copy()
    df = df.rename(columns={"label_tactic": "attack_type"})
    df["label"] = (df["label_binary"] == "True").astype(int)
    df["attack_type"] = df["attack_type"].astype(str)
    df["source"] = dataset_name
    return add_zeek_hybrid_features(df, group_col="source") if with_features else df


def load_unsw_zeek(data_dir: Path | None = None, with_features: bool = True) -> pd.DataFrame:
    """Load the 20 GB UNSW-NB15 Zeek-converted Parquet audit dataset."""
    base = data_dir or (DATASETS_DIR / "unsw_zeek")
    df = pd.read_parquet(base / "unsw_20gb_zeek.parquet")
    df["source"] = "unsw_zeek"
    return add_zeek_hybrid_features(df, group_col="source") if with_features else df


def load_ton_iot(data_dir: Path | None = None) -> pd.DataFrame:
    """Load TON_IoT network CSV with canonical Zeek column names (no timestamps available)."""
    base = data_dir or (DATASETS_DIR / "ton_iot")
    df = pd.read_csv(base / "train_test_network.csv").rename(columns=_COLUMN_ALIASES)
    if "type" in df.columns:
        df = df.rename(columns={"type": "attack_type"})
    return df


def sample_balanced_uwf(df: pd.DataFrame, max_per_class: int = 80_000, seed: int = 42) -> pd.DataFrame:
    """Sample balanced benign/attack rows stratified across tactics so UWF doesn't drown CIC-IDS2017."""
    benign = df[df["label"] == 0]
    attack = df[df["label"] == 1]
    benign_s = benign.sample(n=min(len(benign), max_per_class), random_state=seed)
    if len(attack) <= max_per_class:
        attack_s = attack
    else:
        parts, rem = [], max_per_class
        counts = attack["attack_type"].value_counts().sort_values()
        for i, (tactic, cnt) in enumerate(counts.items()):
            take = min(cnt, max(1, rem // (len(counts) - i)))
            parts.append(attack[attack["attack_type"] == tactic].sample(n=take, random_state=seed))
            rem -= take
        attack_s = pd.concat(parts, ignore_index=True)
    return pd.concat([benign_s, attack_s], ignore_index=True)
