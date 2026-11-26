# CyberShield

LAN-based cybersecurity monitoring combining Zeek `conn.log` ingestion, behavioral feature engineering, and gradient-boosted ML intrusion detection.

## Quick Start

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install Python and dependencies:
   ```sh
   uv sync
   ```
3. Place the Zeek Parquet datasets under `datasets/` (see [datasets/README.md](datasets/README.md)).

## Common Commands

```sh
uv run pytest                               # Run unit tests
uv run python -m cybershield.ml.train_zeek  # Train & save the Zeek Hybrid model -> models/
uv run jupyter lab                          # Open notebooks and reports
```

## Project Layout

| Path | Description |
|---|---|
| [cross_network_zeek_ids_report.ipynb](cross_network_zeek_ids_report.ipynb) | Standalone report evaluating cross-network generalization across public Zeek datasets and AIT-LDSv2 |
| `src/cybershield/ml/` | Zeek feature engineering, dataset loaders (`zeek_features.py`), model training & scoring (`train_zeek.py`) |
| `notebooks/` | Exploratory notebooks ([notebooks/README.md](notebooks/README.md)), `archive/`, and `colab_scripts/` |
| `tests/` | `pytest` unit tests |
| `datasets/` | Dataset documentation and local data directory (gitignored) |
