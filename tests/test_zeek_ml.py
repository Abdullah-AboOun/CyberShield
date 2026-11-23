"""Unit tests for the 19-feature Native Zeek Hybrid feature builder, pipeline, and scoring."""

import numpy as np
import pandas as pd

from cybershield.ml.train_zeek import build_zeek_pipeline, evaluate_host_minutes, evaluate_predictions, score_zeek_flows
from cybershield.ml.zeek_features import ZEEK_HYBRID_FEATURES, add_zeek_hybrid_features, relabel_cic2017


def _sample_zeek_conn_log() -> pd.DataFrame:
    """Synthetic raw Zeek conn.log (using official Zeek JSON column names and '-' missing values)."""
    return pd.DataFrame(
        [
            # Normal host (10.0.0.10) browsing HTTPS + DNS
            {"ts": 1700000001.0, "id.orig_h": "10.0.0.10", "id.resp_h": "1.1.1.1", "id.resp_p": 443, "proto": "tcp", "service": "ssl", "conn_state": "SF", "duration": 1.2, "orig_bytes": 500, "resp_bytes": 4200, "missed_bytes": 0, "orig_pkts": 8, "orig_ip_bytes": 820, "resp_pkts": 10, "resp_ip_bytes": 4600, "label": 0},
            {"ts": 1700000010.0, "id.orig_h": "10.0.0.10", "id.resp_h": "8.8.8.8", "id.resp_p": 53, "proto": "udp", "service": "dns", "conn_state": "SF", "duration": 0.02, "orig_bytes": 60, "resp_bytes": 120, "missed_bytes": 0, "orig_pkts": 1, "orig_ip_bytes": 88, "resp_pkts": 1, "resp_ip_bytes": 148, "label": 0},
            {"ts": 1700000020.0, "id.orig_h": "10.0.0.10", "id.resp_h": "1.1.1.1", "id.resp_p": 443, "proto": "tcp", "service": "ssl", "conn_state": "SF", "duration": 2.0, "orig_bytes": 600, "resp_bytes": 5000, "missed_bytes": 0, "orig_pkts": 10, "orig_ip_bytes": 1000, "resp_pkts": 12, "resp_ip_bytes": 5480, "label": 0},
            # Attacker host (10.0.0.99) port-scanning 3 ports with failed S0/REJ states and '-' nulls
            {"ts": 1700000002.0, "id.orig_h": "10.0.0.99", "id.resp_h": "10.0.0.50", "id.resp_p": 22, "proto": "tcp", "service": "-", "conn_state": "S0", "duration": "-", "orig_bytes": "-", "resp_bytes": "-", "missed_bytes": 0, "orig_pkts": 1, "orig_ip_bytes": 44, "resp_pkts": 0, "resp_ip_bytes": 0, "label": 1},
            {"ts": 1700000003.0, "id.orig_h": "10.0.0.99", "id.resp_h": "10.0.0.50", "id.resp_p": 80, "proto": "tcp", "service": None, "conn_state": "REJ", "duration": 0.001, "orig_bytes": 0, "resp_bytes": 0, "missed_bytes": 0, "orig_pkts": 1, "orig_ip_bytes": 44, "resp_pkts": 1, "resp_ip_bytes": 40, "label": 1},
            {"ts": 1700000004.0, "id.orig_h": "10.0.0.99", "id.resp_h": "10.0.0.50", "id.resp_p": 445, "proto": "tcp", "service": "-", "conn_state": "S0", "duration": 0.0, "orig_bytes": 0, "resp_bytes": 0, "missed_bytes": 0, "orig_pkts": 1, "orig_ip_bytes": 44, "resp_pkts": 0, "resp_ip_bytes": 0, "label": 1},
        ]
    )


def test_zeek_hybrid_features_schema_and_values():
    raw = _sample_zeek_conn_log()
    featured = add_zeek_hybrid_features(raw)

    assert len(ZEEK_HYBRID_FEATURES) == 19
    assert "host_uniq_ips_60s" not in ZEEK_HYBRID_FEATURES
    assert "src_ip" not in ZEEK_HYBRID_FEATURES
    assert "dst_port" not in ZEEK_HYBRID_FEATURES
    assert all(c in featured.columns for c in ZEEK_HYBRID_FEATURES)
    assert not featured[ZEEK_HYBRID_FEATURES].isna().any().any()

    # Attacker host made 3 failed connections across 3 unique ports in the same 60s window
    atk = featured[featured["src_ip"] == "10.0.0.99"]
    assert (atk["host_conn_60s"] == 3).all()
    assert (atk["host_uniq_ports_60s"] == 3).all()
    assert np.allclose(atk["host_failed_ratio_60s"], 1.0)

    # Normal host made 3 successful connections across 2 unique ports (443, 53)
    norm = featured[featured["src_ip"] == "10.0.0.10"]
    assert (norm["host_conn_60s"] == 3).all()
    assert (norm["host_uniq_ports_60s"] == 2).all()
    assert np.allclose(norm["host_failed_ratio_60s"], 0.0)


def test_zeek_pipeline_fit_score_and_allowlist():
    raw = pd.concat([_sample_zeek_conn_log()] * 10, ignore_index=True)
    featured = add_zeek_hybrid_features(raw)

    model = build_zeek_pipeline(max_iter=20)
    model.fit(featured[ZEEK_HYBRID_FEATURES], featured["label"].to_numpy(dtype=int))

    scored = score_zeek_flows(model, _sample_zeek_conn_log(), high_threshold=0.80, suspicious_threshold=0.50)
    assert set(scored["severity"]).issubset({"normal", "suspicious", "high"})
    assert scored.loc[scored["src_ip"] == "10.0.0.99", "attack_prob"].mean() > scored.loc[scored["src_ip"] == "10.0.0.10", "attack_prob"].mean()

    # Allowlisting the attacker IP suppresses its severity back to 'normal'
    allowlisted = score_zeek_flows(model, _sample_zeek_conn_log(), allowlist_src_ips={"10.0.0.99"})
    assert (allowlisted.loc[allowlisted["src_ip"] == "10.0.0.99", "severity"] == "normal").all()


def test_evaluate_predictions_metrics():
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    res = evaluate_predictions(y_true, probs, threshold=0.5)
    assert res["roc_auc"] == 1.0
    assert res["f1"] == 1.0
    assert res["false_positive_rate"] == 0.0


def _cic_ts(hh: int, mm: int) -> float:
    """Epoch seconds for 2017-07-05 hh:mm local CIC time (UTC-3)."""
    return 1499212800.0 + (hh + 3) * 3600 + mm * 60


def test_relabel_cic2017_uses_schedule_ips_and_direction():
    rows = pd.DataFrame(
        [
            # (day, ts, src, dst, old csv label)
            ("wednesday", _cic_ts(10, 50), "172.16.0.1", "192.168.10.50", 1),  # DoS Hulk window
            ("wednesday", _cic_ts(10, 50), "192.168.10.50", "172.16.0.1", 0),  # reply direction counts too
            ("wednesday", _cic_ts(12, 0), "172.16.0.1", "192.168.10.50", 0),  # outside every window
            ("thursday", _cic_ts(15, 10), "192.168.10.8", "192.168.10.5", 0),  # infiltration scan the CSV missed
            ("thursday", _cic_ts(15, 10), "192.168.10.8", "192.168.10.3", 0),  # victim's normal DC traffic
        ],
        columns=["day", "ts", "src_ip", "dst_ip", "label"],
    ).assign(attack_type="x")
    out = relabel_cic2017(rows)
    assert out["label"].tolist() == [1, 1, 0, 1, 0]
    assert out["attack_type"].tolist() == ["DoS Hulk", "DoS Hulk", "BENIGN", "Infiltration - Portscan", "BENIGN"]
    assert out["label_csv"].tolist() == [1, 0, 0, 0, 0]


def test_evaluate_host_minutes_counts_each_host_minute_once():
    # 3 flows of one attacking host-minute (only 1 flagged) + 2 benign host-minutes (1 false alarm)
    df = pd.DataFrame({
        "source": "s", "src_ip": ["a", "a", "a", "b", "c"], "window_id": [1, 1, 1, 1, 1],
        "label": [1, 1, 1, 0, 0],
    })
    res = evaluate_host_minutes(df, np.array([0.9, 0.1, 0.1, 0.8, 0.1]))
    assert res["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 1}
