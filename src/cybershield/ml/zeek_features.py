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

# CIC-IDS2017 ground truth from UNB's published attack schedule (local time, UTC-3), windows widened a few
# minutes to the observed flow times. Replaces the CICFlowMeter CSV labels, which marked e.g. the ~70k-flow
# infiltration port scan from 192.168.10.8 as BENIGN (Engelen et al. 2021 document such errors).
# (day, start "HH:MM", end "HH:MM" inclusive, attack_type, attacker IPs, victim IPs); traffic either way counts.
_CIC_ATTACKER = {"172.16.0.1"}  # 205.174.165.73 as seen inside the LAN after the firewall NAT
_CIC_C2 = {"205.174.165.73"}
_CIC_WEB = {"192.168.10.50"}
# Hosts 192.168.10.8 port-scanned after infiltration (all internal hosts except the DC .3, whose traffic is normal).
_CIC_SCANNED = {f"192.168.10.{i}" for i in (5, 9, 12, 14, 15, 16, 17, 19, 25, 50, 51)}
CIC_ATTACK_SCHEDULE = [
    ("tuesday", "09:17", "10:30", "FTP-Patator", _CIC_ATTACKER, _CIC_WEB),
    ("tuesday", "14:00", "15:15", "SSH-Patator", _CIC_ATTACKER, _CIC_WEB),
    ("wednesday", "09:47", "10:11", "DoS slowloris", _CIC_ATTACKER, _CIC_WEB),
    ("wednesday", "10:14", "10:36", "DoS Slowhttptest", _CIC_ATTACKER, _CIC_WEB),
    ("wednesday", "10:43", "11:01", "DoS Hulk", _CIC_ATTACKER, _CIC_WEB),
    ("wednesday", "11:10", "11:23", "DoS GoldenEye", _CIC_ATTACKER, _CIC_WEB),
    ("wednesday", "15:11", "15:33", "Heartbleed", _CIC_ATTACKER, {"192.168.10.51"}),
    ("thursday", "09:15", "10:01", "Web Attack - Brute Force", _CIC_ATTACKER, _CIC_WEB),
    ("thursday", "10:15", "10:36", "Web Attack - XSS", _CIC_ATTACKER, _CIC_WEB),
    ("thursday", "10:40", "10:43", "Web Attack - Sql Injection", _CIC_ATTACKER, _CIC_WEB),
    ("thursday", "14:19", "15:46", "Infiltration", _CIC_C2, {"192.168.10.8", "192.168.10.25"}),
    ("thursday", "14:33", "15:46", "Infiltration - Portscan", {"192.168.10.8"}, _CIC_SCANNED),
    # ARES bots keep beaconing to the C&C after the 10:02-11:02 session, so the rest of the day counts too.
    ("friday", "10:00", "23:59", "Bot", _CIC_C2, {f"192.168.10.{i}" for i in (5, 8, 9, 14, 15)}),
    ("friday", "13:55", "15:25", "PortScan", _CIC_ATTACKER, _CIC_WEB),
    ("friday", "15:56", "16:17", "DDoS", _CIC_ATTACKER, _CIC_WEB),
]
# AIT scenario config: attacker VPN address and the attacker's dnsteal DNS server.
AIT_ATTACKER_IPS = {"10.9.0.10", "192.168.230.122"}

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


def relabel_cic2017(df: pd.DataFrame) -> pd.DataFrame:
    """Set `label`/`attack_type` from CIC_ATTACK_SCHEDULE; keep the old CSV labels as `label_csv`/`attack_type_csv`."""
    out = df.rename(columns={"label": "label_csv", "attack_type": "attack_type_csv"})
    # CIC was captured 3-7 July 2017 in New Brunswick (ADT, fixed UTC-3): minute of the local day.
    minute = ((pd.to_numeric(out["ts"]) - 3 * 3600) % 86400) // 60
    label = np.zeros(len(out), dtype=np.int64)
    attack_type = np.full(len(out), "BENIGN", dtype=object)
    for day, start, end, name, attackers, victims in CIC_ATTACK_SCHEDULE:
        lo = int(start[:2]) * 60 + int(start[3:])
        hi = int(end[:2]) * 60 + int(end[3:])
        between = (out["src_ip"].isin(attackers) & out["dst_ip"].isin(victims)) | (
            out["src_ip"].isin(victims) & out["dst_ip"].isin(attackers)
        )
        hit = ((out["day"] == day) & minute.between(lo, hi) & between).to_numpy()
        label[hit] = 1
        attack_type[hit] = name
    out["label"] = label
    out["attack_type"] = attack_type
    return out


def load_cic2017_zeek(data_dir: Path | None = None, with_features: bool = True) -> pd.DataFrame:
    """Load the 5-day native Zeek CIC-IDS2017 Parquet files, relabeled from UNB's attack schedule."""
    base = data_dir or (DATASETS_DIR / "cic2017_zeek")
    days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    df = pd.concat([pd.read_parquet(base / f"cic2017_{d}_zeek.parquet") for d in days], ignore_index=True)
    df = relabel_cic2017(df)
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


def load_unsw_official_zeek(data_dir: Path | None = None, with_features: bool = True) -> pd.DataFrame:
    """Load the creators' own UNSW-NB15 Zeek conn.logs, labelled from the official ground-truth table.

    Built by notebooks/colab_scripts/colab_unsw_official_zeek.py from the official UNSW download.
    """
    base = data_dir or (DATASETS_DIR / "unsw_official")
    df = pd.read_parquet(base / "conn.parquet")
    df["source"] = "unsw_official"
    return add_zeek_hybrid_features(df, group_col="source") if with_features else df


def load_ait_zeek(data_dir: Path | None = None, with_features: bool = True) -> pd.DataFrame:
    """Load AIT-LDSv2 russellmitchell Zeek-converted Parquet (deduped, labeled by attacker IPs).

    The PCAPs were recorded on the attacker host during two attack steps, so this is a recall-only test:
    the few benign rows are host noise (IPv6/mDNS/loopback), not real users.
    """
    base = data_dir or (DATASETS_DIR / "ait_zeek")
    df = pd.read_parquet(base / "ait_russellmitchell_zeek.parquet")
    df = df.drop_duplicates(subset=["ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto"]).copy()
    # Either end: dnsteal exfiltration arrives as DNS *to* the attacker, so src-only labels called it benign.
    is_attack = df["src_ip"].isin(AIT_ATTACKER_IPS) | df["dst_ip"].isin(AIT_ATTACKER_IPS)
    df["label"] = is_attack.astype(int)
    is_exfil = df["dst_ip"].isin(AIT_ATTACKER_IPS) & (pd.to_numeric(df["dst_port"], errors="coerce") == 53)
    df["attack_type"] = np.select([is_exfil, is_attack], ["AIT-dnsteal exfil", "AIT-red team"], "BENIGN")
    df["source"] = "ait_russellmitchell"
    return add_zeek_hybrid_features(df, group_col="source") if with_features else df


def load_ait_lds_zeek(
    scenarios: list[str] | None = None, data_dir: Path | None = None, with_features: bool = True
) -> pd.DataFrame:
    """Load the labelled AIT-LDSv2 Zeek conn logs (8 enterprise networks, company capture points only).

    Built by notebooks/colab_scripts/ait_lds_label.py: one row per connection after merging copies seen by
    several probes; attacks matched to the attacker's own capture, exfiltration to the firewall's DNS log.
    Host-context features are computed per scenario (`source`).
    """
    base = data_dir or (DATASETS_DIR / "ait_lds_zeek" / "labelled")
    files = [base / f"{s}.parquet" for s in scenarios] if scenarios else sorted(base.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["source"] = "ait_" + df["scenario"]
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
