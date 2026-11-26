"""Build a labelled Zeek dataset from the OFFICIAL UNSW-NB15 download (run on Google Colab).

Run:  colab new -s unsw && colab exec -s unsw --timeout 900 -f notebooks/colab_scripts/colab_unsw_official_zeek.py
      then `colab download` the files in /content/unsw_official/out into datasets/unsw_official/.

1. Downloads the creators' Zeek (Bro) conn.logs, the ground-truth table and the four full CSVs
   from the official UNSW SharePoint share (https://research.unsw.edu.au/projects/unsw-nb15-dataset).
2. Labels each connection from the ground-truth table: same 5-tuple, start time within the
   attack's [start - 1 s, end + 1 s]. Unmatched connections are normal.
3. Drops the broken Day 1 part 11 log and duplicate rows from overlapping parts.
Outputs (small): conn.parquet, gt.parquet, parts.csv, summary.json.
"""
import glob, json, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SHARE = "https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS"
SITE = "https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au"
ROOT = "/personal/z5025758_ad_unsw_edu_au/Documents/UNSW-NB15 dataset"
RAW, OUT = Path("/content/unsw_official/raw"), Path("/content/unsw_official/out")
RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
BROKEN_PARTS = {("D1", "11")}  # overlaps parts 07-10 in time; 92% single-packet OTH rows
TIME_SLACK = 1  # ground-truth times are whole seconds

s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0"
s.get(SHARE, timeout=60).raise_for_status()
q = lambda p: requests.utils.quote(p.replace("'", "''"))

def ls(folder, kind):
    r = s.get(f"{SITE}/_api/web/GetFolderByServerRelativeUrl('{q(folder)}')/{kind}",
              headers={"Accept": "application/json;odata=nometadata"}, timeout=60)
    r.raise_for_status()
    return r.json()["value"]

def get(path, dest):
    if dest.exists():
        return
    with s.get(f"{SITE}/_api/web/GetFileByServerRelativeUrl('{q(path)}')/$value", stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)

# ---- 1. download ----
t0 = time.time()
bro = f"{ROOT}/BRO Files"
for day in ls(bro, "Folders"):
    tag = "D1" if "22-1" in day["Name"] else "D2"
    for part in ls(f"{bro}/{day['Name']}", "Folders"):
        names = {f["Name"] for f in ls(f"{bro}/{day['Name']}/{part['Name']}", "Files")}
        if "conn.log" in names:
            get(f"{bro}/{day['Name']}/{part['Name']}/conn.log", RAW / f"{tag}_{int(part['Name']):02d}.log")
csv = f"{ROOT}/CSV Files"
for name in ["NUSW-NB15_GT.csv", "NUSW-NB15_features.csv"] + [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]:
    get(f"{csv}/{name}", RAW / name)
print(f"downloaded in {time.time() - t0:.0f}s")

# ---- 2. load conn.logs ----
fields = open(sorted(RAW.glob("D1_*.log"))[0]).read().split("#fields\t")[1].split("\n")[0].split("\t")
ren = {"id.orig_h": "src_ip", "id.orig_p": "src_port", "id.resp_h": "dst_ip", "id.resp_p": "dst_port"}
parts = []
for f in sorted(RAW.glob("D*_*.log")):
    d = pd.read_csv(f, sep="\t", comment="#", names=fields, na_values=["-"], low_memory=False).rename(columns=ren)
    d["day"], d["part"] = f.stem.split("_")
    parts.append(d)
raw = pd.concat(parts, ignore_index=True)

key5 = ["src_ip", "src_port", "dst_ip", "dst_port", "proto"]
part_stats = (raw.groupby(["day", "part"])
              .agg(rows=("uid", "size"), ts_min=("ts", "min"), ts_max=("ts", "max"),
                   oth_share=("conn_state", lambda x: (x == "OTH").mean()))
              .reset_index())
part_stats["repeated_5tuple_share"] = [raw[(raw.day == r.day) & (raw.part == r.part)].duplicated(key5[:4]).mean()
                                       for r in part_stats.itertuples()]
part_stats["kept"] = [(r.day, r.part) not in BROKEN_PARTS for r in part_stats.itertuples()]

# ---- 3. clean: drop the broken part, then rows duplicated across overlapping parts ----
keep = ~pd.Series(list(zip(raw.day, raw.part))).isin(BROKEN_PARTS).to_numpy()
conn = raw[keep].drop_duplicates(["ts"] + key5).reset_index(drop=True)
print(f"raw {len(raw):,} rows -> clean {len(conn):,}")

# ---- 4. ground truth + labelling ----
def num_port(x):
    x = x.astype(str).str.strip()
    hexm = x.str.startswith("0x")
    out = pd.to_numeric(x.where(~hexm), errors="coerce")
    out[hexm] = x[hexm].map(lambda v: int(v, 16))
    return out.fillna(-1).astype(np.int64)

gt = pd.read_csv(RAW / "NUSW-NB15_GT.csv", encoding="latin-1", low_memory=False)
gt.columns = [c.strip() for c in gt.columns]
gt = gt.rename(columns={"Start time": "t0", "Last time": "t1", "Attack category": "attack_type", "Attack Name": "attack_name",
                        "Protocol": "proto", "Source IP": "src_ip", "Source Port": "src_port",
                        "Destination IP": "dst_ip", "Destination Port": "dst_port"})
gt = gt[["t0", "t1", "attack_type", "attack_name", "proto", "src_ip", "src_port", "dst_ip", "dst_port"]].copy()
gt["src_port"], gt["dst_port"] = num_port(gt.src_port), num_port(gt.dst_port)
gt["proto"] = gt.proto.astype(str).str.strip().str.lower()
gt["attack_type"] = gt.attack_type.astype(str).str.strip().replace({"Backdoors": "Backdoor"})
gt["gt_id"] = np.arange(len(gt))

conn["src_port"], conn["dst_port"] = conn.src_port.astype(np.int64), conn.dst_port.astype(np.int64)
conn["row"] = np.arange(len(conn))
m = conn[key5 + ["row", "ts"]].merge(gt[key5 + ["t0", "t1", "attack_type", "gt_id"]], on=key5)
m = m[(m.ts >= m.t0 - TIME_SLACK) & (m.ts <= m.t1 + TIME_SLACK)]
conn["attack_type"] = conn.row.map(m.drop_duplicates("row").set_index("row").attack_type).fillna("Normal")
conn["label"] = (conn.attack_type != "Normal").astype(np.int8)
gt["matched"] = gt.gt_id.isin(m.gt_id)
print(f"attacks labelled: {conn.label.sum():,} ({conn.label.mean():.2%})")

# ---- 5. compare with the creators' own labels in the full CSVs ----
names = [n.strip().lower() for n in pd.read_csv(RAW / "NUSW-NB15_features.csv", encoding="latin-1")["Name"]]
full = pd.concat([pd.read_csv(RAW / f"UNSW-NB15_{i}.csv", header=None, names=names, encoding="latin-1", low_memory=False,
                              usecols=["srcip", "sport", "dstip", "dsport", "proto", "stime", "ltime", "label"])
                  for i in range(1, 5)], ignore_index=True)
full = full.rename(columns={"srcip": "src_ip", "sport": "src_port", "dstip": "dst_ip", "dsport": "dst_port"})
full["src_port"], full["dst_port"] = num_port(full.src_port), num_port(full.dst_port)
fm = conn[key5 + ["row", "ts", "label"]].merge(full[key5 + ["stime", "ltime"] + ["label"]].rename(columns={"label": "csv_label"}), on=key5)
fm = fm[(fm.ts >= fm.stime - TIME_SLACK) & (fm.ts <= fm.ltime + TIME_SLACK)].drop_duplicates("row")

# Current Zeek conn.log fields, for the "same fields as today's Zeek" check
src = requests.get("https://raw.githubusercontent.com/zeek/zeek/master/scripts/base/protocols/conn/main.zeek", timeout=30).text
info = src[src.index("type Info: record"):]; info = info[:info.index("};")]
current = [f for f in re.findall(r"^\s*([a-z_]+)\s*:\s*[^;]*&log", info, re.M) if f not in ("ts", "uid", "id")]

summary = {
    "raw_rows": int(len(raw)), "clean_rows": int(len(conn)),
    "bro_fields": fields, "current_zeek_fields_missing": [f for f in current if f not in fields],
    "full_csv_rows": int(len(full)), "conn_rows_matched_to_full_csv": int(len(fm)),
    "label_agreement_with_full_csv": float((fm.label == fm.csv_label).mean()),
}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

keep_cols = ["ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "duration", "orig_bytes", "resp_bytes",
             "conn_state", "missed_bytes", "history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
             "day", "part", "label", "attack_type"]
conn[keep_cols].to_parquet(OUT / "conn.parquet", index=False, compression="zstd")
gt.drop(columns="gt_id").to_parquet(OUT / "gt.parquet", index=False, compression="zstd")
part_stats.to_csv(OUT / "parts.csv", index=False)
for p in sorted(OUT.iterdir()):
    print(f"{p.name:14s} {p.stat().st_size / 1e6:6.1f} MB")
print(json.dumps({k: v for k, v in summary.items() if k != "bro_fields"}, indent=1))
