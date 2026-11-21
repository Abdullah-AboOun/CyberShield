"""UNSW-NB15 loading and preprocessing (task AI-02).

Turns the raw CSV into model inputs:
- X: feature columns only (no id, no labels)
- y: the binary target `label` (0 = normal, 1 = attack)
Text columns are one-hot encoded. The encoder is fitted on training data only.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path(__file__).resolve().parents[3] / "datasets" / "unsw_nb15"
TRAIN_CSV = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_CSV = DATA_DIR / "UNSW_NB15_testing-set.csv"

TARGET = "label"
# id is a row number; attack_cat and label are answers, so feeding them in would be cheating.
NOT_FEATURES = ["id", "attack_cat", TARGET]
CATEGORICAL = ["proto", "service", "state"]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)  # pandas strips the files' leading BOM itself


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # No imputer by design: filling gaps would fabricate features (spec FR-13), so refuse them.
    empty = df.columns[df.isna().any()].tolist()
    if empty:
        raise ValueError(f"empty values in columns {empty}; fix the data, do not impute")
    return df.drop(columns=NOT_FEATURES), df[TARGET]


def validation_split(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """~80/20 fit/validation split where identical feature rows stay on the same side.

    The training file is ~39% exact duplicates; a plain random split put copies of
    training rows into validation (46.6% of it), which inflated validation scores.
    Grouping by the row's content prevents that; stratification keeps the class ratio.
    """
    groups = pd.util.hash_pandas_object(X, index=False)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(splitter.split(X, y, groups))
    return X.iloc[fit_idx], X.iloc[val_idx], y.iloc[fit_idx], y.iloc[val_idx]


def build_preprocessor(min_frequency: int = 100, categorical: list[str] = CATEGORICAL) -> ColumnTransformer:
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
        [("categorical", encoder, categorical)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
