from __future__ import annotations

from typing import Any, cast

import pandas as pd


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as dicts with str keys; missing values become None."""
    clean = df.astype(object).where(df.notna(), None)
    return cast(list[dict[str, Any]], clean.to_dict("records"))
