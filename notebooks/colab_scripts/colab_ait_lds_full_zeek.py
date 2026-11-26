"""Run Zeek on every pcap of all 8 AIT-LDSv2.0 scenarios (Zenodo record 5789064) on Colab.

It runs for hours, so copy it to the VM and start it detached (nohup), logging to /content/ait_out/progress.log.
It is resumable: finished scenarios are skipped.

Per scenario: download the zip, extract each pcap one at a time (server `log.pcap.<epoch>` files and the
attacker's `traffic.pcap`), run Zeek, keep conn.log, delete the pcap, delete the zip.
Output: /content/ait_out/<scenario>_conn.parquet with a `probe` column (the host that captured it).
The same connection can be captured by several probes, so it may appear more than once.
"""
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import pandas as pd

OUT, WORK = Path("/content/ait_out"), Path("/content/ait_work")
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)
SCENARIOS = os.environ.get("AIT_SCENARIOS", "russellmitchell,fox,harrison,santos,shaw,wardbeck,wheeler,wilson").split(",")
KEEP = ["ts", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service", "duration",
        "orig_bytes", "resp_bytes", "conn_state", "missed_bytes", "history",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
RENAME = {"id.orig_h": "src_ip", "id.orig_p": "src_port", "id.resp_h": "dst_ip", "id.resp_p": "dst_port"}

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

def sh(cmd):
    subprocess.run(["bash", "-c", cmd], check=True)

if not Path("/opt/zeek/bin/zeek").exists():
    log("installing zeek")
    sh("""set -e; V=$(lsb_release -rs)
    echo "deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_${V}/ /" > /etc/apt/sources.list.d/zeek.list
    curl -fsSL "https://download.opensuse.org/repositories/security:/zeek/xUbuntu_${V}/Release.key" | gpg --dearmor > /etc/apt/trusted.gpg.d/zeek.gpg
    apt-get update -qq && apt-get install -y -qq zeek-core""")
sh("command -v aria2c || (apt-get update -qq && apt-get install -y -qq aria2)")
sh("/opt/zeek/bin/zeek --version")

def read_conn(path):
    with open(path) as f:
        fields = next(l for l in f if l.startswith("#fields")).rstrip("\n").split("\t")[1:]
    df = pd.read_csv(path, sep="\t", comment="#", names=fields, usecols=KEEP, na_values=["-", "(empty)"],
                     dtype={"id.orig_h": str, "id.resp_h": str, "history": str, "service": str}, low_memory=False)
    return df.rename(columns=RENAME)

for sc in SCENARIOS:
    out = OUT / f"{sc}_conn.parquet"
    if out.exists():
        log(f"{sc}: done already")
        continue
    zp = WORK / f"{sc}.zip"
    t0 = time.time()
    log(f"{sc}: downloading")
    # 16 parallel connections: a single connection from Zenodo runs at about 3 MB/s
    sh(f"aria2c -q -c -x 16 -s 16 -k 1M --file-allocation=none --max-tries=20 --retry-wait=30 -d {WORK} -o {sc}.zip "
       f"https://zenodo.org/api/records/5789064/files/{sc}.zip/content")
    log(f"{sc}: downloaded {zp.stat().st_size/1e9:.1f} GB in {time.time()-t0:.0f}s")
    zf = zipfile.ZipFile(zp)
    pcaps = [i for i in zf.infolist() if re.search(r"\.pcap(\.\d+)?$", i.filename)]
    listing = pd.DataFrame({"member": [i.filename for i in pcaps], "bytes": [i.file_size for i in pcaps]})
    listing.to_csv(OUT / f"{sc}_pcaps.csv", index=False)
    log(f"{sc}: {len(pcaps)} pcaps, {listing.bytes.sum()/1e9:.1f} GB uncompressed")
    frames = []
    for k, info in enumerate(pcaps, 1):
        m = re.search(r"gather/([^/]+)/", info.filename)
        probe = m.group(1) if m else "unknown"
        local = Path(zf.extract(info, WORK / "x"))
        zdir = WORK / "z"
        sh(f"rm -rf {zdir} && mkdir -p {zdir}")
        r = subprocess.run(["/opt/zeek/bin/zeek", "-C", "-r", str(local)], cwd=zdir, capture_output=True, text=True)
        local.unlink()
        if (zdir / "conn.log").exists():
            c = read_conn(zdir / "conn.log")
            c["probe"], c["pcap_file"] = probe, info.filename
            frames.append(c)
            n = len(c)
        else:
            n = 0
            log(f"  no conn.log for {info.filename}: {r.stderr[-300:]}")
        if k % 10 == 0 or k == len(pcaps):
            log(f"  {sc}: {k}/{len(pcaps)} pcaps, last {probe} {info.file_size/1e6:.0f} MB -> {n:,} conns")
    zf.close()
    zp.unlink()
    sh(f"rm -rf {WORK}/x {WORK}/z")
    df = pd.concat(frames, ignore_index=True)
    for col in ["proto", "service", "conn_state", "probe", "pcap_file"]:
        df[col] = df[col].astype("category")
    df["scenario"] = sc
    df.to_parquet(out, index=False)
    log(f"{sc}: saved {len(df):,} conns, {out.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s total")
log("ALL DONE")
