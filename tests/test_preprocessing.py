import pandas as pd
import pytest

from cybershield.ml.preprocessing import NOT_FEATURES, build_preprocessor, split_xy


def make_rows(protos: list[str]) -> pd.DataFrame:
    n = len(protos)
    return pd.DataFrame({
        "id": range(n),
        "dur": [0.1] * n,
        "proto": protos,
        "service": ["-"] * n,
        "state": ["FIN"] * n,
        "sbytes": [100] * n,
        "attack_cat": ["Normal"] * n,
        "label": [0] * n,
    })


def test_split_xy_removes_id_and_answers():
    X, y = split_xy(make_rows(["tcp", "udp"]))
    assert not set(NOT_FEATURES) & set(X.columns)
    assert y.tolist() == [0, 0]


def test_split_xy_refuses_empty_values():
    df = make_rows(["tcp", "udp"])
    df.loc[0, "sbytes"] = None
    with pytest.raises(ValueError, match="sbytes"):
        split_xy(df)


def test_encoder_learns_from_train_only_and_survives_unseen_categories():
    X_train, _ = split_xy(make_rows(["tcp"] * 5 + ["udp"] * 5 + ["sctp"]))
    X_test, _ = split_xy(make_rows(["tcp", "never_seen"]))

    pre = build_preprocessor(min_frequency=2).fit(X_train)
    out = pre.transform(X_test)

    # tcp/udp are common -> own columns; rare sctp and unseen never_seen share "infrequent".
    assert {"proto_tcp", "proto_udp", "proto_infrequent_sklearn"} <= set(out.columns)
    assert out.loc[1, "proto_infrequent_sklearn"] == 1
    assert out.select_dtypes("number").shape == out.shape  # everything is numeric now
