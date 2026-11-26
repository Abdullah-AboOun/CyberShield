# CyberShield ML Notebooks

Notebooks 01 and 02 test the claims in the dataset evaluation report; 03 tests training across datasets. Earlier exploration is kept in `archive/`.

## Start here
1. **[01_dataset_claims.ipynb](01_dataset_claims.ipynb)**: checks the report's data claims one at a time. Each section states the claim, runs one check, shows one small table or chart, and gives a verdict. No models.
2. **[02_do_the_flaws_matter.ipynb](02_do_the_flaws_matter.ipynb)**: four small experiments on whether those flaws change what a model learns. Every experiment reports the same two numbers: attacks caught and false alarms.
3. **[03_cross_dataset_training.ipynb](03_cross_dataset_training.ipynb)**: trains on some datasets and tests on unseen ones (train on 4 / test on 1, train on 3 / test on 2), with and without the 60-second window features, then tries ways to close the gap.
4. **[04_new_network_test.ipynb](04_new_network_test.ipynb)**: four training sets (UNSW-NB15; CIC-IDS2017; CIC-IDS2017 + UNSW-NB15; CIC-IDS2017 + UWF-2024) tested on unseen networks: per-flow vs. windowed, +1,000 local normal connections, and capped vs. uncapped training.

## Data the notebooks need (in `datasets/`, gitignored)
| Folder | Source |
|---|---|
| `unsw_nb15/` | Official UNSW-NB15 training and testing CSVs |
| `unsw_official/` | UNSW-NB15's own Zeek logs, labelled; built by `colab_scripts/colab_unsw_official_zeek.py` from the official download |
| `uwf_zeek/` | UWF-ZeekData releases from datasets.uwf.edu |
| `cic2017_zeek/` | CIC-IDS2017 converted to Zeek from a HuggingFace mirror (the official download needs registration) |

## Other folders
- `archive/`: earlier experiments (`01`–`10`), including the previous dataset-comparison (`09`) and hybrid-model (`10`) notebooks.
- `colab_scripts/`: Google Colab scripts that turn raw captures and official files into the Parquet files above.
- `samar/`: teammate dataset inspection notebooks.
