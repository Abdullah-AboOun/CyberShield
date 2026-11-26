"""Fetch the small AIT-LDSv2 files needed to label our Zeek data, without downloading the big zips.

Runs locally: HTTP range reads pull single members out of each Zenodo zip (~5 MB per scenario).
Output: datasets/ait_lds_zeek/meta/<scenario>/<original path inside the zip>

    uv run --with remotezip python -I notebooks/colab_scripts/ait_lds_fetch_meta.py [scenario ...]
"""
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteZip

SCENARIOS = sys.argv[1:] or ["fox", "harrison", "russellmitchell", "santos", "shaw", "wardbeck", "wheeler", "wilson"]
OUT = Path(__file__).resolve().parents[2] / "datasets" / "ait_lds_zeek" / "meta"
WANT = re.compile(
    r"^(processing/config/.*\.ya?ml"
    r"|labels/(?!intranet_server/logs/apache2/).*"
    r"|gather/attacker_0/logs/(attacks\.log|dnsteal\.log)"
    r"|gather/attacker_0/configs/dnsteal\.sh"
    r"|gather/inet-firewall/logs/dnsmasq\.log.*"
    r"|gather/vpn/logs/openvpn.*\.log.*)$"
)
MAX_COMPRESSED = 50_000_000  # guard against pulling something huge by accident


def fetch(sc):
    dest = OUT / sc
    with RemoteZip(f"https://zenodo.org/records/5789064/files/{sc}.zip?download=1") as z:
        members = [i for i in z.infolist() if WANT.match(i.filename) and not i.filename.endswith("/")]
        for i in members:
            target = (dest / i.filename).resolve()
            assert target.is_relative_to(dest.resolve()), i.filename  # no path traversal from zip names
            if i.compress_size > MAX_COMPRESSED:
                print(f"{sc}: skip {i.filename} ({i.compress_size:,} B compressed)")
                continue
            if target.exists() and target.stat().st_size == i.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(i))
    return sc, len(members)


with ThreadPoolExecutor(4) as pool:
    for sc, n in pool.map(fetch, SCENARIOS):
        print(f"{sc}: {n} files")
