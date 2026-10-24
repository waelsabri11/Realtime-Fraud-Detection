import numpy as np
import pandas as pd

from fraudrt.monitoring.segments import amount_bucket, segment_report


def test_segment_report_counts_and_rates() -> None:
    df = pd.DataFrame(
        {
            "isFraud": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            "TransactionAmt": [10, 600, 30, 60, 120, 300, 20, 40, 80, 90],
            "g": ["a", "a", "a", "a", "b", "b", "b", "b", None, None],
        }
    )
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.85, 0.05, 0.04, 0.03])
    rep = segment_report(df, score, ["g"], alert_rate=0.4)  # top 4 rows alert: 0, 1, 6, 2
    assert rep["overall"]["alerts"] == 4
    assert rep["overall"]["recall"] == 1.0
    seg = {r["segment"]: r for r in rep["segments"]["g"]}
    assert seg["a"]["alerts"] == 3 and seg["a"]["recall"] == 1.0 and seg["a"]["fpr"] == 0.5
    assert seg["b"]["alerts"] == 1 and seg["b"]["precision"] == 1.0 and seg["b"]["fpr"] == 0.0
    assert seg["missing"]["alerts"] == 0 and seg["missing"]["recall"] is None
    assert rep["segments"]["amount_bucket"][0]["segment"] == "<25"


def test_amount_bucket_edges() -> None:
    out = amount_bucket(pd.Series([24.99, 25.0, 499.99, 500.0]))
    assert list(out) == ["<25", "25-50", "250-500", ">=500"]
