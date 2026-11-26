"""Download AIT-NDS (labelled Tstat flows, Zenodo record 6610489) on Colab and save the columns needed
to label our Zeek conn.logs: 5-tuple, start time, duration, bytes, packets, label.

Output: /content/ait_out/ait_nds_labels.parquet (one row per Tstat flow, all 8 scenarios).
"""
import os
import re
import subprocess
import zipfile

import pandas as pd

os.makedirs("/content/nds", exist_ok=True)
os.makedirs("/content/ait_out", exist_ok=True)
BASE = "https://zenodo.org/records/6610489/files/"
SCENARIOS = ["fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"]
# Attack step names in AIT-NDS; everything else (benign DNS, proxy, HTTPS, mail, ...) is normal activity
ATTACK = re.compile(r"^(host_discover|data exfiltration|service_scan|dirb_scan|dns_brute|online_cracking|wpscan|"
                    r"upload_rce|list_|check_|read_|dump_|vpn_connect|clear|update/command)")
TCP = {"c_ip:1": "src_ip", "c_port:2": "src_port", "s_ip:15": "dst_ip", "s_port:16": "dst_port",
       "first:29": "start_ms", "durat:31": "duration_ms", "c_bytes_all:9": "orig_bytes", "s_bytes_all:23": "resp_bytes",
       "c_pkts_all:3": "orig_pkts", "s_pkts_all:17": "resp_pkts", "label": "nds_label"}
UDP = {"c_ip:1": "src_ip", "c_port:2": "src_port", "s_ip:10": "dst_ip", "s_port:11": "dst_port",
       "c_first_abs:3": "start_ms", "c_durat:4": "duration_ms", "c_bytes_all:5": "orig_bytes", "s_bytes_all:14": "resp_bytes",
       "c_pkts_all:6": "orig_pkts", "s_pkts_all:15": "resp_pkts", "label": "nds_label"}

frames = []
for sc in SCENARIOS:
    zp = f"/content/nds/{sc}_netflows.zip"
    if not os.path.exists(zp):
        subprocess.run(["wget", "-q", f"{BASE}{sc}_netflows.zip?download=1", "-O", zp], check=True)
    z = zipfile.ZipFile(zp)
    for name, cols, proto in [("tcp_complete.csv", TCP, "tcp"), ("tcp_nocomplete.csv", TCP, "tcp"), ("udp_complete.csv", UDP, "udp")]:
        header = z.open(name).readline().decode().strip().lstrip("#").split(",")
        header = [h.split("#")[-1] for h in header]  # tcp headers start with "#15#"
        d = pd.read_csv(z.open(name), skiprows=1, names=header, usecols=list(cols), low_memory=False).rename(columns=cols)
        d["proto"], d["nds_file"], d["scenario"] = proto, name.removesuffix(".csv"), sc
        frames.append(d)
    print(sc, "ok", flush=True)

df = pd.concat(frames, ignore_index=True)
df["nds_label"] = df.nds_label.astype(str)
df["label"] = df.nds_label.str.match(ATTACK).astype("int8")
for c in ["nds_label", "proto", "nds_file", "scenario"]:
    df[c] = df[c].astype("category")
df.to_parquet("/content/ait_out/ait_nds_labels.parquet", index=False)
print(len(df), "flows;", int(df.label.sum()), "attacks;", os.path.getsize("/content/ait_out/ait_nds_labels.parquet") // 1_000_000, "MB")
print(df.groupby("scenario", observed=True).label.agg(["size", "sum"]))
