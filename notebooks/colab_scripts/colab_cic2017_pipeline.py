"""Run on Google Colab Pro via `colab --auth=oauth2 exec -s cybershield -f notebooks/colab_cic2017_pipeline.py`.

1. Installs Zeek on the Colab VM.
2. Converts all 5 days of CIC-IDS2017 (Mon-Fri) PCAPs -> native Zeek conn.log -> labeled Parquet.
3. Compares Pure Row-by-Row (11 Zeek features) vs Hybrid (11 Zeek + 60s Host Context) across all 5 days.
4. Saves the trained Hybrid model to /content/cic2017_zeek/hybrid_zeek_cic2017.joblib.
"""

from pathlib import Path
import gc
import shutil
import subprocess
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

OUT_DIR = Path("/content/cic2017_zeek")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main"

DAYS = {
    "monday": (
        "pcap/Monday-WorkingHours.pcap",
        ["traffic_labels/Monday-WorkingHours.pcap_ISCX.csv.parquet"],
    ),
    "tuesday": (
        "pcap/Tuesday-WorkingHours.pcap",
        ["traffic_labels/Tuesday-WorkingHours.pcap_ISCX.csv.parquet"],
    ),
    "wednesday": (
        "pcap/Wednesday-workingHours.pcap",
        ["traffic_labels/Wednesday-workingHours.pcap_ISCX.csv.parquet"],
    ),
    "thursday": (
        "pcap/Thursday-WorkingHours.pcap",
        [
            "traffic_labels/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
            "traffic_labels/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet",
        ],
    ),
    "friday": (
        "pcap/Friday-WorkingHours.pcap",
        [
            "traffic_labels/Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
            "traffic_labels/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
            "traffic_labels/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
        ],
    ),
}

# 1. Install Zeek if not present
if not Path("/opt/zeek/bin/zeek").exists():
    print("=== Installing Zeek on Colab VM ===", flush=True)
    cmd = """
    set -e
    UBUNTU_VER=$(lsb_release -rs)
    echo "deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_${UBUNTU_VER}/ /" > /etc/apt/sources.list.d/security:zeek.list
    curl -fsSL "https://download.opensuse.org/repositories/security:/zeek/xUbuntu_${UBUNTU_VER}/Release.key" | gpg --dearmor > /etc/apt/trusted.gpg.d/security_zeek.gpg
    apt-get update -qq
    apt-get install -y -qq zeek-core
    /opt/zeek/bin/zeek --version
    """
    subprocess.run(["bash", "-c", cmd], check=True)
else:
    print("Zeek already installed.", flush=True)

# 2. Convert all 5 days to labeled Zeek Parquet
KEEP_COLS = [
    "ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "conn_state",
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "label", "attack_type",
]

for day, (pcap_rel, lbl_rels) in DAYS.items():
    out_path = OUT_DIR / f"cic2017_{day}_zeek.parquet"
    if out_path.exists():
        print(f"[{day}] Already exists: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", flush=True)
        continue

    t0 = time.time()
    work = Path(f"/content/work_{day}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    pcap_local = work / "day.pcap"

    print(f"\n=== [{day.upper()}] Downloading PCAP ===", flush=True)
    subprocess.run(["curl", "-fsSL", "-o", str(pcap_local), f"{BASE_URL}/{pcap_rel}"], check=True)
    print(f"[{day}] Downloaded {pcap_local.stat().st_size/1e9:.2f} GB in {time.time()-t0:.0f}s. Running Zeek...", flush=True)

    zeek_dir = work / "zeek"
    zeek_dir.mkdir()
    subprocess.run(
        ["/opt/zeek/bin/zeek", "-C", "-r", str(pcap_local), "LogAscii::use_json=T"],
        cwd=zeek_dir,
        check=True,
    )
    pcap_local.unlink()  # free disk immediately

    conn = pd.read_json(zeek_dir / "conn.log", lines=True).rename(columns={
        "id.orig_h": "src_ip",
        "id.orig_p": "src_port",
        "id.resp_h": "dst_ip",
        "id.resp_p": "dst_port",
    })

    lbl_dfs = []
    for rel in lbl_rels:
        lp = work / Path(rel).name
        subprocess.run(["curl", "-fsSL", "-o", str(lp), f"{BASE_URL}/{rel}"], check=True)
        lbl_dfs.append(pd.read_parquet(lp))
    lbl = pd.concat(lbl_dfs, ignore_index=True)
    lbl.columns = [c.strip() for c in lbl.columns]
    lbl["proto"] = lbl["Protocol"].map({6: "tcp", 17: "udp", 1: "icmp"}).fillna("other")
    lbl["is_attack"] = (lbl["Label"].astype(str).str.upper() != "BENIGN").astype(int)

    lbl_key = (
        lbl.groupby(["Source IP", "Source Port", "Destination IP", "Destination Port", "proto"], sort=False)
        .agg(
            label=("is_attack", "max"),
            attack_type=("Label", lambda s: next((str(x) for x in s if str(x).upper() != "BENIGN"), "BENIGN")),
        )
        .reset_index()
        .rename(columns={
            "Source IP": "src_ip",
            "Source Port": "src_port",
            "Destination IP": "dst_ip",
            "Destination Port": "dst_port",
        })
    )

    joined = conn.merge(lbl_key, on=["src_ip", "src_port", "dst_ip", "dst_port", "proto"], how="inner")
    joined["day"] = day
    cols = [c for c in KEEP_COLS + ["day"] if c in joined.columns]
    joined[cols].to_parquet(out_path, index=False)
    print(
        f"[{day}] Done in {time.time()-t0:.0f}s | matched {len(joined):,}/{len(conn):,} "
        f"({len(joined)/max(len(conn),1)*100:.1f}%) | saved {out_path.stat().st_size/1e6:.1f} MB",
        flush=True,
    )
    print("  Labels:", joined["attack_type"].value_counts().to_dict(), flush=True)

    del conn, lbl, lbl_key, joined
    gc.collect()
    shutil.rmtree(work)

# 3. Load all 5 days and build Hybrid features
print("\n=== LOADING ALL 5 DAYS & BUILDING HYBRID FEATURES ===", flush=True)
dfs = [pd.read_parquet(OUT_DIR / f"cic2017_{d}_zeek.parquet") for d in DAYS]
df = pd.concat(dfs, ignore_index=True)
del dfs
gc.collect()

CAT_COLS = ["proto", "service", "conn_state"]
RAW_NUM = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]
FAILED_STATES = {"S0", "REJ", "RSTO", "RSTR", "SH", "SHR", "RSTOS0", "RSTRH"}

for c in CAT_COLS:
    df[c] = df[c].fillna("-").astype(str).astype("category")
for c in RAW_NUM:
    df[c] = df[c].fillna(0.0).astype(np.float32)

df["orig_bytes_per_pkt"] = (df["orig_bytes"] / df["orig_pkts"].clip(lower=1.0)).astype(np.float32)
df["resp_bytes_per_pkt"] = (df["resp_bytes"] / df["resp_pkts"].clip(lower=1.0)).astype(np.float32)
df["byte_ratio"] = (df["orig_bytes"] / (df["orig_bytes"] + df["resp_bytes"] + 1.0)).astype(np.float32)

df["is_failed"] = df["conn_state"].astype(str).isin(FAILED_STATES).astype(np.float32)
df["window_id"] = (df["ts"] // 60).astype("int64")

host_ctx = df.groupby(["day", "src_ip", "window_id"], sort=False).agg(
    host_conn_60s=("ts", "size"),
    host_uniq_ips_60s=("dst_ip", "nunique"),
    host_uniq_ports_60s=("dst_port", "nunique"),
    host_failed_ratio_60s=("is_failed", "mean"),
    host_orig_bytes_60s=("orig_bytes", "sum"),
    host_resp_bytes_60s=("resp_bytes", "sum"),
).reset_index()
host_ctx["host_ports_per_ip_60s"] = (
    host_ctx["host_uniq_ports_60s"] / host_ctx["host_uniq_ips_60s"].clip(lower=1)
).astype(np.float32)
for c in ["host_conn_60s", "host_uniq_ips_60s", "host_uniq_ports_60s", "host_failed_ratio_60s", "host_orig_bytes_60s", "host_resp_bytes_60s"]:
    host_ctx[c] = host_ctx[c].astype(np.float32)

df = df.merge(host_ctx, on=["day", "src_ip", "window_id"], how="left")
del host_ctx
gc.collect()

CTX_COLS = [
    "orig_bytes_per_pkt", "resp_bytes_per_pkt", "byte_ratio",
    "host_conn_60s", "host_uniq_ips_60s", "host_uniq_ports_60s",
    "host_ports_per_ip_60s", "host_failed_ratio_60s",
    "host_orig_bytes_60s", "host_resp_bytes_60s",
]
RAW_FEATS = CAT_COLS + RAW_NUM
HYBRID_FEATS = RAW_FEATS + CTX_COLS

print(f"Total 5-day Zeek flows: {len(df):,} | Benign={(df['label']==0).sum():,} | Attack={(df['label']==1).sum():,}", flush=True)
print("Attack breakdown:", df["attack_type"].value_counts().to_dict(), flush=True)

# 4. Leak-free grouped 80/20 split by (day, src_ip, window_id)
groups = df["day"].astype(str) + "_" + df["src_ip"].astype(str) + "_" + df["window_id"].astype(str)
y = df["label"].to_numpy(dtype=np.int8)
t = df["attack_type"].to_numpy(dtype=str)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, te_idx = next(sgkf.split(df, y, groups))
print(f"\nGrouped split | train={len(tr_idx):,} | test={len(te_idx):,}", flush=True)


def run_eval(name, feat_cols):
    num_cols = [c for c in feat_cols if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(random_state=42, max_iter=200)),
    ])
    X_tr = df.iloc[tr_idx][feat_cols].copy()
    X_te = df.iloc[te_idx][feat_cols].copy()
    for c in CAT_COLS:
        X_tr[c] = X_tr[c].astype(object)
        X_te[c] = X_te[c].astype(object)
    pipe.fit(X_tr, y[tr_idx])
    probs = pipe.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[te_idx], preds).ravel()
    print(f"\n================ {name} ================", flush=True)
    print(
        f"Overall | F1={f1_score(y[te_idx], preds):.4f} | Prec={precision_score(y[te_idx], preds):.4f} "
        f"| Rec={recall_score(y[te_idx], preds):.4f} | FPR={fp/(fp+tn):.5f}",
        flush=True,
    )
    print("Per-attack breakdown:", flush=True)
    for atk in sorted(np.unique(t[te_idx])):
        m = t[te_idx] == atk
        rate = (preds[m] == 1).mean()
        lbl = "FPR" if atk.upper() == "BENIGN" else "recall"
        print(f"  {atk:28s} (n={m.sum():7,}): {lbl}={rate:.4f}", flush=True)
    return pipe, X_te


m_raw, _ = run_eval("1. Pure Row-by-Row (11 Zeek features, 5 days)", RAW_FEATS)
m_hyb, X_te_hyb = run_eval("2. Hybrid (11 Zeek + 60s Host Context, 5 days)", HYBRID_FEATS)

# Permutation importance on Hybrid (25k sample for speed)
sub_n = min(25_000, len(X_te_hyb))
sub_idx = np.random.default_rng(42).choice(len(X_te_hyb), size=sub_n, replace=False)
imp = permutation_importance(
    m_hyb, X_te_hyb.iloc[sub_idx], y[te_idx][sub_idx],
    n_repeats=3, random_state=42, scoring="f1", n_jobs=-1,
)
print("\nTop Hybrid features (F1 drop when shuffled):", flush=True)
for idx in np.argsort(imp.importances_mean)[::-1][:12]:
    print(f"  {HYBRID_FEATS[idx]:22s}: {imp.importances_mean[idx]:.4f}", flush=True)

# Save trained Hybrid model
model_path = OUT_DIR / "hybrid_zeek_cic2017.joblib"
joblib.dump({"model": m_hyb, "features": HYBRID_FEATS, "cat_cols": CAT_COLS}, model_path)
print(f"\nSaved Hybrid model to {model_path} ({model_path.stat().st_size/1e6:.2f} MB)", flush=True)
