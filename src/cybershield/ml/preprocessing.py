"""UNSW-NB15 loading and preprocessing (task AI-02).

Turns the raw CSV into model inputs:
- X: feature columns only (no id, no labels)
- y: the binary target `label` (0 = normal, 1 = attack)
Text columns are one-hot encoded. The encoder is fitted on training data only.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path(__file__).resolve().parents[3] / "datasets" / "unsw_nb15"
TRAIN_CSV = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_CSV = DATA_DIR / "UNSW_NB15_testing-set.csv"

TARGET = "label"
# id is a row number; attack_cat and label are answers, so feeding them in would be cheating.
NOT_FEATURES = ["id", "attack_cat", TARGET]
CATEGORICAL = ["proto", "service", "state"]


def load_csv(path: Path) -> pd.DataFrame:
    # utf-8-sig strips the BOM at the start of the official files (else column 1 is "﻿id").
    return pd.read_csv(path, encoding="utf-8-sig")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=NOT_FEATURES), df[TARGET]


def build_preprocessor(min_frequency: int = 100) -> ColumnTransformer:
    """Unfitted transformer: one-hot encode text columns, pass numeric columns through unchanged.

    Categories seen fewer than `min_frequency` times in training are grouped into one
    "infrequent" column; categories never seen in training (e.g. in test or Zeek data) go there too.
    """
    encoder = OneHotEncoder(
        min_frequency=min_frequency,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )
    return ColumnTransformer(
        [("categorical", encoder, CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
