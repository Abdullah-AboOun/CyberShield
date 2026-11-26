"""Compare our AIT-LDSv2 Zeek labels (ait_lds_label.py) with AIT-NDS labels (Tstat flows).

Match: same (proto, src_port, dst_ip, dst_port), start within 1 s (src_ip ignored because of NAT).
NDS attack/normal follows AIT's official label_info.txt (NDS v2), not our regex.

    uv run python -I notebooks/colab_scripts/ait_lds_crosscheck.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2] / "datasets" / "ait_lds_zeek"
NDS_ATTACK = {  # "Malicious labels" in AIT-NDS v2 label_info.txt
    "service_scan", "online_cracking", "host_discover_dmz", "host_discover_local", "wpscan", "dirb_scan",
    "upload_rce_shell", "check_user_id", "check_netstat_t", "read_resolv", "check_network_config", "check_ps_a",
    "check_release", "read_group", "read_passwd", "check_date", "list_web_dir", "check_wp_config", "dump_wp_users",
    "read_profile", "dns_brute_force_start", "list_www", "check_who", "clear", "check_last", "check_id", "vpn_connect",
    "check_whoami", "check_uname_r", "check_meminfo", "check_uname_a", "check_df", "check_netstat_nat", "list_home",
    "check_netstat_l", "check_cpuinfo", "check_uptime", "check_pwd", "list_l", "data exfiltration",
}
KEY = ["proto", "src_port", "dst_ip", "dst_port"]

nds = pd.read_parquet(BASE / "ait_nds_labels.parquet")
for c in nds.select_dtypes("category"):
    nds[c] = nds[c].astype(str)
nds["ts"] = nds.start_ms / 1000
nds["nds_attack"] = nds.nds_label.isin(NDS_ATTACK).astype(int)
nds = nds.sort_values("nds_attack", ascending=False).drop_duplicates(KEY + ["scenario", "start_ms"])

rows, mism = [], []
for f in sorted((BASE / "labelled").glob("*.parquet")):
    ours = pd.read_parquet(f)
    sc = ours.scenario.iloc[0]
    n = nds[nds.scenario == sc][KEY + ["ts", "nds_label", "nds_attack"]].sort_values("ts")
    for c in ("src_port", "dst_port"):
        n[c] = n[c].astype("int64")
    m = pd.merge_asof(ours.sort_values("ts"), n, on="ts", by=KEY, tolerance=1.0, direction="nearest")
    hit = m.nds_label.notna()
    ct = pd.crosstab(m[hit].label, m[hit].nds_attack)
    rows.append({
        "scenario": sc, "ours": len(m), "matched_to_nds": f"{hit.mean():.0%}",
        "ours_attack": int(m.label.sum()), "both_attack": int(ct.loc[1, 1]) if 1 in ct.index and 1 in ct.columns else 0,
        "ours_attack_nds_normal": int(ct.loc[1, 0]) if 1 in ct.index and 0 in ct.columns else 0,
        "ours_normal_nds_attack": int(ct.loc[0, 1]) if 0 in ct.index and 1 in ct.columns else 0,
    })
    d = m[hit & (m.label != m.nds_attack)]
    mism.append(d.groupby(["label", "attack_type", "nds_label"]).size().rename("n").reset_index().assign(scenario=sc))

print(pd.DataFrame(rows).to_string(index=False))
mm = pd.concat(mism).groupby(["label", "attack_type", "nds_label"]).n.sum().sort_values(ascending=False)
print("\nDisagreements (label = ours):\n", mm.head(25).to_string())
