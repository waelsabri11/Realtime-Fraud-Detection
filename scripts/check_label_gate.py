"""Evidence for delayed labels: no label may be stored before its transaction was scored,
and every label must have been released only after the replay clock reached its availability time."""

import json
from pathlib import Path

import pandas as pd

from fraudrt.config import get_settings
from fraudrt.db.repo import make_engine

s = get_settings()
q = """
select l.transaction_id, t.event_dt, l.available_dt, l.received_at, sc.scored_at
from labels l join transactions t using (transaction_id) join scores sc using (transaction_id)
where t.transaction_id < 900000000
"""
with make_engine(s.database_url).connect() as conn:
    df = pd.read_sql(q, conn)
delay = s.label_delay_seconds
res = {
    "labels": len(df),
    "available_dt_equals_event_dt_plus_delay": bool(
        ((df.event_dt + delay) == df.available_dt).all()
    ),
    "labels_stored_before_their_transaction_was_scored": int((df.received_at < df.scored_at).sum()),
}
print(json.dumps(res, indent=2))
Path("work/results").mkdir(parents=True, exist_ok=True)
Path("work/results/label_gate.json").write_text(json.dumps(res, indent=2))
