# CyberShield ML Notebooks — 60-Second Guide

Only **2 clean, presentation-ready notebooks** live here at the top level. All earlier exploration and dead ends are preserved in `archive/`.

## Start Here
1. **[01_dataset_comparison.ipynb](01_dataset_comparison.ipynb)** — Compares all 4 dataset families we evaluated (`CIC-IDS2017` Zeek, `UWF-ZeekData`, `TON_IoT`, `UNSW-NB15` Zeek), with sources, pros/cons, and why `UNSW-NB15`'s 2015 IXIA traffic is excluded from host-window training.
2. **[02_zeek_hybrid_model.ipynb](02_zeek_hybrid_model.ipynb)** — Shows how our production **19-feature Native Zeek Hybrid model (`zeek-hybrid-m3-v1`)** beats pure per-flow detection, fixes the `host_uniq_ips_60s` lab shortcut, and generalizes to unseen 2025 `UWF` captures.

## TL;DR of What Won (and Why)
- **Production model:** `zeek-hybrid-m3-v1` (`HistGradientBoostingClassifier`, trained via `uv run python -m cybershield.ml.train_zeek`).
- **Features (19):** 11 native Zeek `conn.log` fields + 3 flow ratios + 5 per-host 60-second window stats (`window_id = int(ts // 60)`, grouped by `(src_ip, window_id)`, then attached back to each flow row). Raw IPs, ports, and timestamps are never fed to the model.
- **Training data:** `CIC-IDS2017` (5-day native Zeek, real benign TCP/SSL/HTTP/SSH) + balanced chunks of `UWF-ZeekData24` & `UWF-ZeekDataFall24-2`.
- **Scores:** `0.994` F1 / `0.31%` FPR on held-out `CIC-IDS2017`; `0.887–0.999` F1 / `0.001%–0.056%` FPR on unseen 2025 `UWF-Sum25-1/2`; **`0.985` F1 / `0.013%` FPR zero-shot on unseen 2022 `AIT-LDSv2` (`russellmitchell`)**.

## Other Folders
- `archive/` — Earlier chronological experiments (`01`–`08`: legacy UNSW-NB15 baseline, TTL audit, early UWF & pure-window drafts).
- `colab_scripts/` — Reproducible Google Colab scripts used to convert raw `CIC-IDS2017` (52 GB) and `UNSW-NB15` (20 GB) PCAPs into native Zeek Parquet files.
- `samar/` — Teammate dataset inspection notebooks.
