# BRR CyberShield AI — Phase 1

LAN-based cybersecurity monitoring combining Zeek log ingestion, rule-based detection, a Native Zeek Hybrid ML classifier (`zeek-hybrid-m3-v1`), and a local AI assistant.

## Quick Start

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install Python and dependencies:
   ```sh
   uv sync
   ```
3. Put the datasets under `datasets/` (`CIC-IDS2017` & `UWF` Zeek Parquet for training; `AIT-LDSv2`, `UNSW-NB15`, `TON_IoT` for audits). See [datasets/README.md](datasets/README.md).

## Common Commands

```sh
uv run pytest                               # Run test suite (21 tests)
uv run python -m cybershield.ml.train_zeek  # Train & evaluate 19-feature Zeek Hybrid model -> models/
uv run jupyter lab                          # Open notebooks/
```

## Project Layout

| Path | Description |
|---|---|
| `src/cybershield/ml/` | `zeek_features.py` (19 Hybrid features + dataset loaders), `train_zeek.py` (LODO eval, training & scoring), and legacy UNSW baseline (`preprocessing.py`, `train.py`) |
| `notebooks/` | 4 notebooks ([notebooks/README.md](notebooks/README.md)), `archive/`, `colab_scripts/`, and `samar/` |
| `tests/` | `pytest` unit tests (`test_zeek_ml.py` + legacy tests) |
| `datasets/` | Dataset comparison, sources, and local data directory (gitignored) |

## Notebooks ([notebooks/README.md](notebooks/README.md))

| Notebook | What it shows |
|---|---|
| [01_dataset_claims.ipynb](notebooks/01_dataset_claims.ipynb) | Tests the dataset evaluation report's data claims one by one (UNSW-NB15, UWF-ZeekData, CIC-IDS2017), each with a verdict |
| [02_do_the_flaws_matter.ipynb](notebooks/02_do_the_flaws_matter.ipynb) | Four small experiments: TTL shortcut, duplicates, shortcuts in UNSW-NB15's Zeek logs, and whether a UNSW-trained model works on another network |
| [03_cross_dataset_training.ipynb](notebooks/03_cross_dataset_training.ipynb) | Train on 4 datasets / test on the 5th and train on 3 / test on 2, per-flow vs. 60s-windowed features, and what closes the cross-network gap |
| [04_new_network_test.ipynb](notebooks/04_new_network_test.ipynb) | Four training sets tested on unseen networks: per-flow vs. windowed, +1,000 local normal connections, capped vs. uncapped training |
| `notebooks/archive/` | Earlier exploration (`01` → `10`), including the previous dataset-comparison and hybrid-model notebooks (`09`, `10`) |

## ML Findings & Honest Benchmarks (`zeek-hybrid-m3-v1`)

Trained on `CIC-IDS2017` (5 days of Zeek, labeled from UNB's official attack schedule) + balanced `UWF-ZeekData24` & `UWF-ZeekDataFall24-2` using `HistGradientBoostingClassifier` on **19 native Zeek `conn.log` + 60s per-host features** (no IPs, no ports, no TTL; threshold `0.50`).

### 3-Line Conclusion
1. **Strong on known, loud network attacks:** `94–100%` recall on DoS (Hulk/Slowloris/Slowhttptest), DDoS, PortScan, FTP-Patator, Web Brute Force/XSS, and `99.7%` on unseen `AIT-LDSv2` red-team scans/exploits—with **5–30× fewer host-minute false alarms** than pure per-flow scoring.
2. **Weak on unseen or stealthy attack types:** Misses brand-new attack families (`SSH-Patator 0.6%`, `GoldenEye 4.7%`, `Bot 5.5%`) and payload/DNS stealth attacks (`SQLi 0%`, `dnsteal 0%`).
3. **Why CyberShield is layered:** Supervised ML is a **known-attack detector**, not a standalone shield—pair it with deterministic Zeek rules, a busy-server allowlist (NAT gateways/DNS), an unsupervised LAN anomaly baseline, and real-LAN FPR calibration.

### CIC-IDS2017 — Leave-One-Day-Out (Honest Unseen-Day Evaluation)
Each attack day is scored by a model that never saw that day *(an earlier random host-minute split leaked adjacent minutes of the same attack and reported an inflated `0.994` F1)*:

| Left-out day | Flow Recall | Flow FPR | Host-minute Recall | Host-minute FPR | Unseen attack types missed |
|---|---:|---:|---:|---:|---|
| Tuesday | 56.9% | 0.03% | 52.8% | 0.49% | SSH-Patator (0.6%) |
| Wednesday | 90.8% | 0.12% | 97.1% | 0.81% | DoS GoldenEye (4.7%), Heartbleed (0%) |
| Thursday | 98.0% | 1.10% | 70.7% | 0.43% | SQL Injection (0%, 12 flows) |
| Friday | 97.7% | 0.005% | 7.9% | 0.07% | Bot (5.5%) — beacons all day, dominating host-minutes |

### Unseen Datasets (Final Model Trained on All 5 CIC Days + UWF)

| Test Set | Flow Recall | Flow FPR | Host-minute Recall | Honest Note |
|---|---:|---:|---:|---|
| `UWF-ZeekDataSum25-1` (2025) | 15.0% | 0% | 30.6% | Its "Recon" is ~6k *successful* conns/min to 3 ports, unlike training; unstable (swung 15–80% across minor changes) |
| `UWF-ZeekDataSum25-2` (2025) | 99.97% | 0.001% | 31 of 32 | All 934k attack flows come from 32 host-minutes |
| `AIT-LDSv2` `russellmitchell` (2022) | 91.8% | — | 1.1% | **Recall-only** (attacker-host PCAPs): red-team scans/exploits **99.7%**, **dnsteal DNS exfil 0%** |

> [!WARNING]
> - Flow-level scores are dominated by floods (one scan = 100k+ flows); always read **host-minute** and **per-attack-type** recall.
> - Benign FPR is only measured on lab traffic; real FPR requires capturing Zeek logs on a real LAN.
> - Live alert tiers (`suspicious ≥ 0.60`, `high ≥ 0.90`) are not yet validated on real LAN traffic.

### Historical Legacy Baseline (`UNSW-NB15` CSV — Argus/TTL)
Kept in `src/cybershield/ml/train.py` (`uv run python -m cybershield.ml.train`) for reference only: Random Forest (`0.894` F1, `26.8%` FPR, 111 MB) and HGB (`0.894` F1, `26.3%` FPR, 1.5 MB). Both rely on Argus/TTL features absent from live Zeek `conn.log`.
