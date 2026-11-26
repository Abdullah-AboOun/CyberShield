"""Label and deduplicate the AIT-LDSv2 Zeek conn logs (runs locally, no pcaps needed).

Inputs (datasets/ait_lds_zeek/): <scenario>_conn.parquet (Zeek on every pcap) and meta/<scenario>/ (fetched by
ait_lds_fetch_meta.py). Output: labelled/<scenario>.parquet, one row per connection, company probes only.

Rules (why: AIT-NDS labels by time window only, which mislabels VPN employees; see docs in datasets/README.md):
  1. Drop the attacker_0 probe from the data; use it as ground truth instead. A company-probe connection is an
     attack if the attacker's own capture has the same (proto, src_port, dst_ip, dst_port) within 1 s during the
     attack session. The source IP is ignored because the VPN server NATs the attacker to its own address.
  2. Any connection to/from the attacker's internet IP during the attack session is an attack.
  3. DNS exfiltration: connections matching a firewall dnsmasq log line for the exfil domain (client -> firewall
     query, firewall -> upstream forward) by IP and second, plus anything to the attacker's IP on port 53.
  4. Trim to the simulation window, then merge copies of one connection seen by several probes.

    uv run python -I notebooks/colab_scripts/ait_lds_label.py [scenario ...]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE = Path(__file__).resolve().parents[2] / "datasets" / "ait_lds_zeek"
# Simulation windows, paper Table 6 (Landauer et al. 2022), UTC
SIM = {
    "fox": ("2022-01-15", "2022-01-20"), "harrison": ("2022-02-04", "2022-02-09"),
    "russellmitchell": ("2022-01-21", "2022-01-25"), "santos": ("2022-01-14", "2022-01-18"),
    "shaw": ("2022-01-25", "2022-01-31"), "wardbeck": ("2022-01-19", "2022-01-24"),
    "wheeler": ("2022-01-26", "2022-01-31"), "wilson": ("2022-02-03", "2022-02-09"),
}
SLACK = 60  # seconds added around the attack session
TOL = 1.0  # seconds, timestamp tolerance between probes


def epoch(t):
    """Seconds since 1970 for a datetime Series, independent of pandas' datetime unit (us vs ns)."""
    return (t - pd.Timestamp("1970-01-01", tz=t.dt.tz)) / pd.Timedelta(seconds=1)


def ts(s):
    return pd.Timestamp(s).timestamp()


def load_meta(sc):
    m = BASE / "meta" / sc
    att = yaml.safe_load((m / "processing/config/attacker/attacker.yaml").read_text())
    exf = yaml.safe_load((m / "processing/config/attacker.yaml").read_text())
    steps = pd.read_csv(m / "gather/attacker_0/logs/attacks.log", sep=" ", header=None, names=["d", "t", "step"])
    steps["ts"] = epoch(pd.to_datetime(steps.d + " " + steps.t, utc=True))
    session = (steps.ts[steps.step == "vpn_connect"].min() - SLACK, steps.ts[steps.step == "vpn_disconnect"].max() + SLACK)
    return {
        "attacker_ip": att["default_ipv4_address"],
        "steps": steps[steps.step != "dnsteal_stop"][["ts", "step"]].sort_values("ts"),
        "session": session,
        "exfil": (ts(exf["dnsteal"]["start"]) - SLACK, ts(exf["dnsteal"]["stop"]) + SLACK),
        "domain": exf["dnsteal"]["domain"],
        "dnsmasq": sorted((m / "gather/inet-firewall/logs").glob("dnsmasq.log*")),
    }


def dnsmasq_exfil(files, domain, year):
    """Return (second, ip) pairs: queries 'from ip' and forwards 'to ip' for the exfil domain."""
    pat = re.compile(r"^(\w{3} +\d+ [\d:]+) dnsmasq\[\d+\]: (?:query\[\w+\] \S+ from|forwarded \S+ to) (\S+)$")
    rows = []
    for f in files:
        for line in f.open(errors="replace"):
            if domain in line and (mt := pat.match(line.rstrip("\n"))):
                rows.append((mt[1], mt[2]))
    df = pd.DataFrame(rows, columns=["t", "ip"])
    df["sec"] = epoch(pd.to_datetime(f"{year} " + df.t, format="%Y %b %d %H:%M:%S")).astype("int64")
    return set(zip(df.sec, df.ip))


def oracle_match(company, oracle):
    """True where a company row has a twin in the attacker capture: same (proto, src_port, dst_ip, dst_port), |dt| <= TOL."""
    key = ["proto", "src_port", "dst_ip", "dst_port"]
    # multicast/broadcast use fixed ports (mDNS 5353, DHCP 67/68), so every host's packets would match the attacker's
    company = company[~company.dst_ip.str.match(r"^(22[4-9]\.|23\d\.|255\.255\.255\.255$|ff0)")]
    left = company[key + ["ts"]].reset_index().sort_values("ts")
    right = oracle[key + ["ts"]].drop_duplicates().sort_values("ts").assign(hit=True)
    m = pd.merge_asof(left, right, on="ts", by=key, tolerance=TOL, direction="nearest")
    return m.set_index("index").hit.astype("boolean").fillna(False).astype(bool)


def dedup(df):
    """Merge copies of one connection seen by several probes (NAT-aware): same proto, src_port, dst_port, start within
    TOL, sharing src_ip or dst_ip with the previous row. Keep the copy with most packets; attack if any copy is."""
    df = df.sort_values(["proto", "src_port", "dst_port", "ts"]).reset_index(drop=True)
    p = df.shift()
    same = (
        (df.proto == p.proto) & (df.src_port == p.src_port) & (df.dst_port == p.dst_port)
        & (df.ts - p.ts <= TOL) & ((df.src_ip == p.src_ip) | (df.dst_ip == p.dst_ip))
    )
    df["cid"] = (~same).cumsum()
    g = df.groupby("cid")
    df["label"] = g.label.transform("max")
    df["attack_type"] = g.attack_type.transform(lambda s: s[s != "normal"].iloc[0] if (s != "normal").any() else "normal")
    df["n_probes"] = g.probe.transform("nunique")
    df["_pk"] = df.orig_pkts + df.resp_pkts
    keep = df.sort_values(["cid", "_pk"], ascending=[True, False]).drop_duplicates("cid")
    return keep.drop(columns=["cid", "_pk"]).sort_values("ts").reset_index(drop=True)


def label(sc):
    meta = load_meta(sc)
    df = pd.read_parquet(BASE / f"{sc}_conn.parquet")
    for c in ["proto", "probe", "pcap_file"]:
        df[c] = df[c].astype(str)
    lo, hi = (pd.Timestamp(x, tz="UTC").timestamp() for x in SIM[sc])
    df = df[(df.ts >= lo) & (df.ts < hi)]
    oracle = df[df.probe == "attacker_0"]
    df = df[df.probe != "attacker_0"].copy()
    n_raw = len(df)

    s0, s1 = meta["session"]
    in_session = df.ts.between(s0, s1)
    ip = meta["attacker_ip"]
    r1 = in_session & oracle_match(df, oracle[oracle.ts.between(s0, s1)]).reindex(df.index, fill_value=False)
    r2 = in_session & ((df.src_ip == ip) | (df.dst_ip == ip)) & ~r1

    e0, e1 = meta["exfil"]
    pairs = dnsmasq_exfil(meta["dnsmasq"], meta["domain"], SIM[sc][0][:4])
    dns = (df.proto == "udp") & (df.dst_port == 53) & df.ts.between(e0, e1)
    sec = df.ts.astype("int64")
    hit = pd.Series(False, index=df.index)
    for d in (-1, 0, 1):  # dnsmasq logs whole seconds; allow clock/rounding offset
        hit |= pd.Series([(s, a) in pairs or (s, b) in pairs for s, a, b in zip(sec + d, df.src_ip, df.dst_ip)], index=df.index)
    r3 = dns & (hit | (df.dst_ip == ip)) & ~r1 & ~r2

    df["label"] = (r1 | r2 | r3).astype(int)
    step = pd.merge_asof(df[["ts"]].reset_index().sort_values("ts"), meta["steps"], on="ts", direction="backward")
    df["attack_type"] = "normal"
    df.loc[r1 | r2, "attack_type"] = step.set_index("index").step.reindex(df.index)[r1 | r2].fillna("vpn_connect")
    df.loc[r3, "attack_type"] = "dns_exfiltration"

    out = dedup(df)
    out["scenario"] = sc
    out.to_parquet(BASE / "labelled" / f"{sc}.parquet", index=False)
    stats = {
        "scenario": sc, "rows_company": n_raw, "rows_dedup": len(out),
        "rule1_oracle": int(r1.sum()), "rule2_attacker_ip": int(r2.sum()), "rule3_exfil": int(r3.sum()),
        "exfil_log_lines": len(pairs), "attacks_dedup": int(out.label.sum()),
    }
    print(stats, flush=True)
    return stats


if __name__ == "__main__":
    (BASE / "labelled").mkdir(exist_ok=True)
    scs = sys.argv[1:] or list(SIM)
    pd.DataFrame([label(s) for s in scs]).to_csv(BASE / "labelled" / "summary.csv", index=False)
