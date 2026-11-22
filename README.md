# BRR CyberShield AI — Phase 1

LAN-based cybersecurity monitoring combining Zeek log ingestion, rule-based detection, a UNSW-NB15 ML classifier, and a local AI assistant.

## Quick Start

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install Python and dependencies:
   ```sh
   uv sync
   ```
3. Download the UNSW-NB15 CSVs into `datasets/unsw_nb15/` (see [datasets/README.md](datasets/README.md)).

## Common Commands

```sh
uv run pytest                          # Run test suite
uv run python -m cybershield.ml.train  # Train & evaluate models -> models/
uv run jupyter lab                     # Open notebooks/
```

## Project Layout

| Path | Description |
|---|---|
| `src/cybershield/` | Core application packages (`ml/` contains the model pipeline) |
| `notebooks/` | Data exploration & model selection (`01` → `11`), Colab PCAP→Zeek scripts (`colab_scripts/`), and `samar/` |
| `tests/` | `pytest` test suite |
| `datasets/` | Local dataset instructions and data directory (gitignored) |

## Notebook Progression (`notebooks/`)

| Notebooks | Focus | Key Outcome |
|---|---|---|
| `01` → `06` | `UNSW-NB15` baseline, threshold tuning, TTL removal & dedup | Exposed TTL/Argus mismatch vs. live Zeek `conn.log` |
| `07` → `08` | Multi-dataset Zeek (`TON_IoT` + `UWF`) & pure 60s host windows | Proved multi-dataset gains; caught UWF idle-lab benign TCP (0 B) |
| `09` → `10` | 5-day `CIC-IDS2017` Zeek conversion, Hybrid features & cross-network test | Selected **19-feature Hybrid `M3`** (removed `host_uniq_ips_60s` shortcut) |
| `11` | Master comparison of all 7 local datasets + `UNSW-NB15` 20 GB Zeek audit | Documented UNSW IXIA normal-traffic inversion (134 vs. 32 conn/min) |

## ML Benchmarks

### Legacy Baseline (`UNSW-NB15` Test Split — Argus/TTL)

| Model | F1 | Recall | False-Positive Rate | Size |
|---|---:|---:|---:|---:|
| Random Forest (baseline) | 0.894 | 98.6% | 26.8% | 111 MB |
| HistGradientBoosting | 0.894 | 98.1% | 26.3% | 1.5 MB |

> [!NOTE]
> False-positive rate jumps from ~7% on validation to ~26% on the official test split due to distribution shift. Both models lean heavily on TTL/Argus features that Zeek `conn.log` does not provide.

### Native Zeek Hybrid Benchmark (`M3` — 19 Zeek `conn.log` + 60s Host-Context Features)

Trained on `CIC-IDS2017` (5-day native Zeek, labeled via UNB official GT schedule) + `UWF-ZeekData24` + `UWF-ZeekDataFall24-2` using `HistGradientBoostingClassifier` (no IPs, no ports, no TTL):

| Test Set | ROC-AUC | F1 | Recall | False-Positive Rate |
|---|---:|---:|---:|---:|
| `CIC-IDS2017` (held-out grouped split) | 0.9999 | **0.9943** | 99.9% | **0.31%** |
| `UWF-ZeekDataSum25-1` (unseen 2025) | 0.9999 | **0.8865** | 79.6% | **0.001%** |
| `UWF-ZeekDataSum25-2` (unseen 2025) | 0.9998 | **0.9991** | 99.9% | **0.056%** |

> [!TIP]
> See [11_all_datasets_comparison.ipynb](notebooks/11_all_datasets_comparison.ipynb) for the full 7-dataset audit and why `UNSW-NB15` synthetic IXIA traffic is excluded from host-window training.
