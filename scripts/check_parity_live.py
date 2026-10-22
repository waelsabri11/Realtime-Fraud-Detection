"""System-level parity: features stored by the live API vs the offline table, per replayed txn."""

import json
import sys

import numpy as np
import pandas as pd

from fraudrt.config import get_settings
from fraudrt.data import load_features
from fraudrt.db.repo import make_engine
from fraudrt.features.definitions import FEATURE_COLUMNS, NUMERIC_FEATURES

s = get_settings()
off = load_features(s).set_index("TransactionID")
with make_engine(s.database_url).connect() as conn:
    db = pd.read_sql("select transaction_id, fraud_score, features::text f from scores", conn)
served = pd.DataFrame([json.loads(x) for x in db["f"]], index=db["transaction_id"])
served = served.reindex(columns=FEATURE_COLUMNS)
ref = off.loc[served.index, FEATURE_COLUMNS]
num = list(NUMERIC_FEATURES)
a, b = served[num].to_numpy(float), ref[num].to_numpy(float)
ok = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
cat = [c for c in FEATURE_COLUMNS if c not in num]
cok = (served[cat].fillna("<na>") == ref[cat].fillna("<na>")).to_numpy()
res = {
    "n_transactions": len(served),
    "numeric_cells_match": float(ok.mean()),
    "categorical_cells_match": float(cok.mean()),
    "rows_fully_matching": float((ok.all(axis=1) & cok.all(axis=1)).mean()),
}
print(json.dumps(res, indent=2))
sys.exit(0 if res["rows_fully_matching"] == 1.0 else 1)
