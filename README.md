# BRR CyberShield AI — Phase 1

LAN-based cybersecurity monitoring with Zeek log ingestion, rule-based detection,
an ML classifier trained on UNSW-NB15, and a local AI assistant.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. `uv sync` (installs Python and all dependencies into `.venv`).
3. Download the UNSW-NB15 training/testing CSVs into `datasets/unsw_nb15/` (see [datasets/README.md](datasets/README.md)).

## Run

```sh
uv run pytest                          # tests
uv run python -m cybershield.ml.train  # train + evaluate models -> models/
uv run jupyter lab                     # notebooks/
```

## Layout

- `src/cybershield/` — application packages (`ml/` is the model pipeline).
- `notebooks/` — data exploration and model selection (01 → 03); `notebooks/samar/` — Samar's dataset inspection.
- `tests/` — pytest suite.

## Current ML results (UNSW-NB15 official test split)

| Model | F1 | Recall | False-positive rate | Size |
|---|---|---|---|---|
| Random Forest (baseline) | 0.894 | 98.6% | 26.8% | 111 MB |
| HistGradientBoosting | 0.894 | 98.1% | 26.3% | 1.5 MB |

False-positive rate is ~7% on a validation split but ~26% on test (distribution shift).
The models rely heavily on TTL features, which Zeek `conn.log` does not provide.
These scores are not evidence of performance on Zeek or real network traffic.
