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
| `notebooks/` | Data exploration & model selection (`01` → `03`); `samar/` for dataset inspection |
| `tests/` | `pytest` test suite |
| `datasets/` | Local dataset instructions and data directory |

## ML Baseline (UNSW-NB15 Test Split)

| Model | F1 | Recall | False-Positive Rate | Size |
|---|---:|---:|---:|---:|
| Random Forest (baseline) | 0.894 | 98.6% | 26.8% | 111 MB |
| HistGradientBoosting | 0.894 | 98.1% | 26.3% | 1.5 MB |

> [!NOTE]
> False-positive rate jumps from ~7% on validation to ~26% on the official test split due to distribution shift. Both models also lean heavily on TTL features that Zeek `conn.log` does not provide, so these scores do not reflect live Zeek performance.
