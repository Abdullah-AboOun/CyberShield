"""Run on Google Colab via `colab --auth=oauth2 exec -s cybershield_unsw --timeout 1800 -f notebooks/colab_unsw_zeek_pipeline.py`.

1. Installs Zeek on the Colab VM.
2. Downloads the full 2.54M-row UNSW-NB15 ground-truth table (with 5-tuple + Stime/Ltime + attack_cat + label).
3. Downloads Mireu-Lab/UNSW-NB15 Wireshark-File.zip (11.2 GB) and extracts 20 PCAPs (~20 GB: 10 from Day 1 + 10 from Day 2).
4. Verifies timestamps & UNSW IPs on the first PCAP to confirm authenticity.
5. Runs Zeek on each PCAP, joins labels strictly by 5-tuple + [Stime-2s, Ltime+2s] timestamp window, and saves compact Parquet.
"""

from pathlib import Path
import gc
import shutil
import subprocess
import time
import zipfile
import numpy as np
import pandas as pd

OUT_DIR = Path("/content/unsw_zeek")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORK = Path("/content/work_unsw")
WORK.mkdir(parents=True, exist_ok=True)

# 1. Install Zeek if needed
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

# 2. Download full 2.54M-row UNSW-NB15 ground-truth table (with Stime, Ltime, 5-tuple, attack_cat, label)
print("\n=== Downloading UNSW-NB15 full ground-truth Parquets (218 MB) ===", flush=True)
gt_parts = []
gt_base = "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/data"
for fname in ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]:
    p = WORK / fname
    if not p.exists():
        subprocess.run(["curl", "-fsSL", "-o", str(p), f"{gt_base}/{fname}"], check=True)
    gt_parts.append(pd.read_parquet(p, columns=["srcip", "sport", "dstip", "dsport", "proto", "Stime", "Ltime", "attack_cat", "label"]))

gt = pd.concat(gt_parts, ignore_index=True)
del gt_parts
gc.collect()

gt["src_ip"] = gt["srcip"].astype(str).str.strip()
gt["dst_ip"] = gt["dstip"].astype(str).str.strip()
gt["src_port"] = pd.to_numeric(gt["sport"], errors="coerce").fillna(-1).astype(np.int32)
gt["dst_port"] = pd.to_numeric(gt["dsport"], errors="coerce").fillna(-1).astype(np.int32)
gt["proto"] = gt["proto"].astype(str).str.strip().str.lower()
gt["Stime"] = pd.to_numeric(gt["Stime"], errors="coerce").fillna(0).astype(np.int64)
gt["Ltime"] = pd.to_numeric(gt["Ltime"], errors="coerce").fillna(0).astype(np.int64)
gt["label"] = pd.to_numeric(gt["label"], errors="coerce").fillna(0).astype(np.int8)
gt["attack_cat"] = gt["attack_cat"].fillna("Normal").astype(str).str.strip()
gt.loc[gt["attack_cat"].isin(["", "None", "nan"]), "attack_cat"] = "Normal"
# Normalize known UNSW-NB15 naming typo ("Backdoors" vs "Backdoor")
gt.loc[gt["attack_cat"] == "Backdoors", "attack_cat"] = "Backdoor"

print(f"Loaded GT table: {len(gt):,} rows | Attacks={(gt['label']==1).sum():,}", flush=True)
print("GT time range (UTC):", pd.to_datetime(gt["Stime"].min(), unit="s"), "->", pd.to_datetime(gt["Ltime"].max(), unit="s"), flush=True)
print("GT attack_cat counts:", gt["attack_cat"].value_counts().to_dict(), flush=True)

# Keep only attack GT rows indexed by 5-tuple for fast time-window matching!
# (In UNSW-NB15, any flow NOT matching an attack in [Stime-2, Ltime+2] is Normal.)
gt_atk = gt[gt["label"] == 1][["src_ip", "src_port", "dst_ip", "dst_port", "proto", "Stime", "Ltime", "attack_cat"]].copy()
# Also build a 5-tuple -> list of (Stime, Ltime, attack_cat) dict for fast O(1) lookup
from collections import defaultdict
atk_index = defaultdict(list)
for row in gt_atk.itertuples(index=False):
    atk_index[(row.src_ip, row.src_port, row.dst_ip, row.dst_port, row.proto)].append((row.Stime - 2, row.Ltime + 2, row.attack_cat))
print(f"Indexed {len(gt_atk):,} attack GT rows across {len(atk_index):,} unique 5-tuples.", flush=True)
del gt, gt_atk
gc.collect()

# 3. Download Mireu-Lab Wireshark-File.zip (11.2 GB)
zip_path = WORK / "Wireshark-File.zip"
if not zip_path.exists():
    t0 = time.time()
    print("\n=== Downloading UNSW-NB15 Wireshark-File.zip (11.2 GB) ===", flush=True)
    subprocess.run([
        "curl", "-fsSL", "-o", str(zip_path),
        "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/Archives/Wireshark-File.zip",
    ], check=True)
    print(f"Downloaded {zip_path.stat().st_size/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)

# Pick 20 PCAPs (~20 GB uncompressed): 10 evenly spaced from 1..27 (Day 1) + 10 evenly spaced from 28..52 (Day 2)
SELECTED_IDS = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 31, 34, 37, 40, 43, 46, 48, 50, 52]
SELECTED_PCAPS = [f"Wireshark-File/{i}.pcap" for i in SELECTED_IDS]

KEEP_COLS = [
    "ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "conn_state",
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "label", "attack_type", "pcap_id",
]

all_parts = []
with zipfile.ZipFile(zip_path, "r") as zf:
    avail = set(zf.namelist())
    to_run = [p for p in SELECTED_PCAPS if p in avail]
    print(f"\n=== Processing {len(to_run)} PCAPs (~20 GB uncompressed) ===", flush=True)

    for idx, member in enumerate(to_run, 1):
        t0 = time.time()
        pcap_id = Path(member).stem
        pcap_local = WORK / f"{pcap_id}.pcap"
        zf.extract(member, path=WORK)
        extracted = WORK / member
        extracted.rename(pcap_local)

        zeek_dir = WORK / f"zeek_{pcap_id}"
        zeek_dir.mkdir(exist_ok=True)
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
        shutil.rmtree(zeek_dir)

        if idx == 1:
            print("\n=== AUTHENTICITY CHECK ON FIRST PCAP ===", flush=True)
            print("  Timestamp range (UTC):", pd.to_datetime(conn["ts"].min(), unit="s"), "->", pd.to_datetime(conn["ts"].max(), unit="s"), flush=True)
            print("  Top src_ip:", conn["src_ip"].value_counts().head(5).to_dict(), flush=True)
            print("  Top dst_ip:", conn["dst_ip"].value_counts().head(5).to_dict(), flush=True)

        # Match against UNSW GT using 5-tuple + [Stime-2, Ltime+2] timestamp window
        labels = np.zeros(len(conn), dtype=np.int8)
        atks = np.full(len(conn), "Normal", dtype=object)

        src_ips = conn["src_ip"].astype(str).to_numpy()
        src_ports = conn["src_port"].fillna(-1).astype(np.int32).to_numpy()
        dst_ips = conn["dst_ip"].astype(str).to_numpy()
        dst_ports = conn["dst_port"].fillna(-1).astype(np.int32).to_numpy()
        protos = conn["proto"].astype(str).str.lower().to_numpy()
        tss = conn["ts"].to_numpy()

        for i in range(len(conn)):
            key = (src_ips[i], src_ports[i], dst_ips[i], dst_ports[i], protos[i])
            hits = atk_index.get(key)
            if hits:
                t_val = tss[i]
                for s_t, e_t, cat in hits:
                    if s_t <= t_val <= e_t:
                        labels[i] = 1
                        atks[i] = cat
                        break
            else:
                # Also check reverse direction in case Zeek saw responder first
                rkey = (dst_ips[i], dst_ports[i], src_ips[i], src_ports[i], protos[i])
                rhits = atk_index.get(rkey)
                if rhits:
                    t_val = tss[i]
                    for s_t, e_t, cat in rhits:
                        if s_t <= t_val <= e_t:
                            labels[i] = 1
                            atks[i] = cat
                            break

        conn["label"] = labels
        conn["attack_type"] = atks
        conn["pcap_id"] = int(pcap_id)
        cols = [c for c in KEEP_COLS if c in conn.columns]
        all_parts.append(conn[cols])

        n_atk = int(labels.sum())
        dt_s = pd.to_datetime(conn["ts"].min(), unit="s").strftime("%Y-%m-%d %H:%M")
        print(
            f"[{idx:2d}/{len(to_run)}] pcap {pcap_id:>2s} ({dt_s} UTC) | flows={len(conn):7,} | "
            f"attacks={n_atk:6,} ({n_atk/max(len(conn),1)*100:4.1f}%) | {time.time()-t0:.0f}s",
            flush=True,
        )

# Free the 11.2 GB zip
zip_path.unlink()

final_df = pd.concat(all_parts, ignore_index=True)
out_path = OUT_DIR / "unsw_20gb_zeek.parquet"
final_df.to_parquet(out_path, index=False)

print(f"\n=== DONE! Saved {out_path} ({out_path.stat().st_size/1e6:.1f} MB) ===", flush=True)
print(f"Total flows: {len(final_df):,} | Normal={(final_df['label']==0).sum():,} | Attack={(final_df['label']==1).sum():,}", flush=True)
print("Attack breakdown:", final_df["attack_type"].value_counts().to_dict(), flush=True)
