"""Run on Google Colab via `colab --auth=oauth2 exec -s cybershield_ait --timeout 1800 -f notebooks/colab_scripts/colab_ait_zeek_pipeline.py`.

1. Installs Zeek on Colab.
2. Downloads AIT-LDSv2 `russellmitchell.zip` (7.13 GB) from Zenodo.
3. Inspects its structure (PCAPs + attack metadata/labels), runs Zeek on the PCAP(s),
   labels attack flows from AIT's own attacker IP/metadata, and saves compact Parquet.
"""

from pathlib import Path
import gc
import json
import re
import shutil
import subprocess
import time
import zipfile
import numpy as np
import pandas as pd

OUT_DIR = Path("/content/ait_zeek")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORK = Path("/content/work_ait")
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

# 2. Download russellmitchell.zip (7.13 GB) from Zenodo
zip_path = WORK / "russellmitchell.zip"
if not zip_path.exists():
    t0 = time.time()
    print("\n=== Downloading AIT-LDSv2 russellmitchell.zip (7.13 GB) ===", flush=True)
    subprocess.run([
        "curl", "-fsSL", "-o", str(zip_path),
        "https://zenodo.org/api/records/5789064/files/russellmitchell.zip/content",
    ], check=True)
    print(f"Downloaded {zip_path.stat().st_size/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)

# 3. Inspect zip contents: find PCAPs, metadata, and label files
with zipfile.ZipFile(zip_path, "r") as zf:
    names = zf.namelist()
    pcaps = [n for n in names if n.lower().endswith((".pcap", ".pcapng"))]
    meta_files = [n for n in names if any(k in n.lower() for k in ["attack", "label", "config", "info", "readme", "servers", "hosts"]) and not n.endswith("/")]
    print(f"\nZip entries: {len(names)} | PCAPs: {len(pcaps)}", flush=True)
    for p in pcaps[:20]:
        info = zf.getinfo(p)
        print(f"  PCAP: {info.file_size/1e6:7.1f} MB  {p}", flush=True)
    print("\nMetadata/config/label candidates (first 25):", flush=True)
    for m in meta_files[:25]:
        info = zf.getinfo(m)
        print(f"  META: {info.file_size/1e3:7.1f} KB  {m}", flush=True)

    # Read small config/yaml/json/txt files that mention attacker IP or attack schedule
    attacker_ips = set()
    attack_windows = []
    for m in names:
        if m.endswith((".yaml", ".yml", ".json", ".txt", ".md")) and zf.getinfo(m).file_size < 200_000:
            if any(k in m.lower() for k in ["attack", "config", "scenario", "info", "readme", "servers", "dns"]):
                txt = zf.read(m).decode("utf-8", "replace")
                print(f"\n--- {m} (preview) ---\n{txt[:600]}", flush=True)

    # Also check Suricata eve.json / fast.log if present, or extract the largest/gateway PCAPs
    # Prefer inet-firewall or intranet_server / largest PCAPs (cap at ~8 GB total)
    pcap_infos = sorted([(zf.getinfo(p).file_size, p) for p in pcaps], reverse=True)
    chosen = []
    for sz, p in pcap_infos:
        if "firewall" in p.lower() or "gateway" in p.lower() or "router" in p.lower() or "ext" in p.lower():
            chosen.append((sz, p))
    if not chosen:
        chosen = pcap_infos[:3]
    print("\nChosen PCAPs to run Zeek on:", [(round(sz/1e6, 1), p) for sz, p in chosen], flush=True)

    all_conns = []
    for sz, p in chosen:
        t0 = time.time()
        zf.extract(p, path=WORK)
        pcap_local = WORK / p
        zeek_dir = WORK / f"zeek_{Path(p).stem}"
        zeek_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["/opt/zeek/bin/zeek", "-C", "-r", str(pcap_local), "LogAscii::use_json=T"],
            cwd=zeek_dir,
            check=True,
        )
        pcap_local.unlink()
        conn_log = zeek_dir / "conn.log"
        if conn_log.exists():
            c = pd.read_json(conn_log, lines=True).rename(columns={
                "id.orig_h": "src_ip", "id.orig_p": "src_port",
                "id.resp_h": "dst_ip", "id.resp_p": "dst_port",
            })
            c["pcap_file"] = p
            all_conns.append(c)
            print(f"  Zeek on {p}: {len(c):,} flows in {time.time()-t0:.0f}s", flush=True)
        shutil.rmtree(zeek_dir)

    # Also extract labels/ folder (small) so we can inspect exact ground-truth attacker IPs & times
    label_members = [n for n in names if n.startswith("labels/") or "/labels/" in n or "attack" in n.lower()]
    for lm in label_members[:20]:
        if not lm.endswith("/") and zf.getinfo(lm).file_size < 50_000_000:
            zf.extract(lm, path=WORK)

zip_path.unlink()

if all_conns:
    df = pd.concat(all_conns, ignore_index=True)
    keep = [c for c in [
        "ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "conn_state",
        "duration", "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "pcap_file",
    ] if c in df.columns]
    out_path = OUT_DIR / "ait_russellmitchell_zeek.parquet"
    df[keep].to_parquet(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df):,} flows, {out_path.stat().st_size/1e6:.1f} MB)", flush=True)
    print("Time range (UTC):", pd.to_datetime(df["ts"].min(), unit="s"), "->", pd.to_datetime(df["ts"].max(), unit="s"), flush=True)
    print("Top src_ip:", df["src_ip"].value_counts().head(10).to_dict(), flush=True)
    print("Top dst_ip:", df["dst_ip"].value_counts().head(10).to_dict(), flush=True)
else:
    print("\nNo PCAPs yielded conn.log!", flush=True)
