# CyberShield ML Notebooks — 60-Second Guide

Only **2 clean, presentation-ready notebooks** live here at the top level. All earlier exploration and dead ends are preserved in `archive/`.

## Start Here
1. **[01_dataset_comparison.ipynb](01_dataset_comparison.ipynb)** — Compares 5 datasets (`CIC-IDS2017`, `UWF-ZeekData`, `AIT-LDSv2`, `TON_IoT`, `UNSW-NB15`), audits and fixes their labels, and scores the production model on every test set.
2. **[02_zeek_hybrid_model.ipynb](02_zeek_hybrid_model.ipynb)** — Per-flow vs. **19-feature Native Zeek Hybrid model (`zeek-hybrid-m3-v1`)** under leave-one-day-out, recall per attack type, and what the model relies on.

## TL;DR
- **Production model:** `zeek-hybrid-m3-v1` (`HistGradientBoostingClassifier`, trained via `uv run python -m cybershield.ml.train_zeek`).
- **Features (19):** 11 native Zeek `conn.log` fields + 3 flow ratios + 5 per-host 60-second window stats (`window_id = int(ts // 60)`, grouped by `(src_ip, window_id)`, then attached back to each flow row). Raw IPs, ports, and timestamps are never fed to the model.
- **Training data:** all 5 `CIC-IDS2017` days (labels from UNB's attack schedule) + balanced chunks of `UWF-ZeekData24` & `UWF-ZeekDataFall24-2`.
- **How it's scored:** CIC leave-one-day-out (attack types the model never saw), per host-minute and per attack type, plus unseen `UWF-Sum25-1/2` and `AIT-LDSv2` (recall-only). Numbers: [README](../README.md#ml-findings--honest-benchmarks-zeek-hybrid-m3-v1).
- **Honest summary:** strong on attack types it trained on (DoS, DDoS, scans, FTP brute force, web brute force/XSS); weak on new ones (SSH-Patator, GoldenEye, Bot, SQLi, DNS exfiltration).

## Other Folders
- `archive/` — Earlier chronological experiments (`01`–`08`: legacy UNSW-NB15 baseline, TTL audit, early UWF & pure-window drafts). Numbers there use the old labels and splits.
- `colab_scripts/` — Google Colab scripts that convert raw `CIC-IDS2017` (52 GB), `UNSW-NB15` (20 GB) and `AIT-LDSv2` (7.1 GB) PCAPs into Zeek Parquet files.
- `samar/` — Teammate dataset inspection notebooks.
