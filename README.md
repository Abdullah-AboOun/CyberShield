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
uv run pytest                               # Run test suite
uv run python -m cybershield.ml.train_zeek  # Train & evaluate 19-feature Zeek Hybrid model -> models/
uv run python -m cybershield.ml.train       # Legacy UNSW-NB15 baseline -> models/
uv run jupyter lab                          # Open notebooks/
```

## Project Layout

| Path | Description |
|---|---|
| `src/cybershield/` | Core application packages (`ml/zeek_features.py` & `ml/train_zeek.py` for Native Zeek Hybrid ML) |
| `notebooks/` | 2 clean presentation notebooks (`01_dataset_comparison`, `02_zeek_hybrid_model`), `archive/`, `colab_scripts/`, and `samar/` |
| `tests/` | `pytest` test suite |
| `datasets/` | Local dataset instructions, pros/cons comparison, and data directory (gitignored) |

## Notebooks ([notebooks/README.md](notebooks/README.md))

| Notebook | Focus |
|---|---|
| [01_dataset_comparison.ipynb](notebooks/01_dataset_comparison.ipynb) | Side-by-side comparison of all 4 dataset families (`CIC-IDS2017`, `UWF`, `TON_IoT`, `UNSW-NB15`), sources, pros/cons & UNSW IXIA inversion audit |
| [02_zeek_hybrid_model.ipynb](notebooks/02_zeek_hybrid_model.ipynb) | Pure per-flow vs. 19-feature Hybrid `M3`, per-attack recall, and feature importance audit |
| `notebooks/archive/` | Earlier chronological experiments (`01` → `08`: legacy UNSW-NB15 baseline, TTL audit, early UWF drafts) |

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
| `AIT-LDSv2` `russellmitchell` (unseen 2022 3rd network, zero-shot) | **0.9990** | **0.9850** | **97.0%** | **0.013%** |

> [!TIP]
> See [01_dataset_comparison.ipynb](notebooks/01_dataset_comparison.ipynb) for the full 7-dataset audit and why `UNSW-NB15` synthetic IXIA traffic is excluded from host-window training.
