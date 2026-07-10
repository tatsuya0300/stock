"""価格データ品質チェック。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_PRICE_COLS = [
    "code",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "volume",
    "turnover",
]

_NUMERIC_COLS = [
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "volume",
    "turnover",
]


def validate_prices(
    df: pd.DataFrame,
    strict: bool = False,
) -> pd.DataFrame:
    """価格データを検証する。

    検証項目:
      - 必須列
      - code/dateの妥当性
      - 数値列のNaN/inf
      - OHLCの整合性
      - adjusted OHLCの整合性
      - volume/turnoverの非負性
      - code/date重複
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_PRICE_COLS)

    missing = [c for c in REQUIRED_PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df[REQUIRED_PRICE_COLS].copy()
    n_before = len(x)

    x["code"] = x["code"].astype("string").str.strip()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")

    for col in _NUMERIC_COLS:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    numeric_values = x[_NUMERIC_COLS].to_numpy(dtype=float)
    finite_mask = pd.Series(
        np.isfinite(numeric_values).all(axis=1),
        index=x.index,
    )

    mask = x["code"].notna() & x["code"].ne("")
    mask &= x["date"].notna()
    mask &= finite_mask

    # raw OHLC
    mask &= x["open"] > 0
    mask &= x["high"] > 0
    mask &= x["low"] > 0
    mask &= x["close"] > 0
    mask &= x["high"] >= x[["open", "close", "low"]].max(axis=1)
    mask &= x["low"] <= x[["open", "close", "high"]].min(axis=1)

    # adjusted OHLC
    mask &= x["adj_open"] > 0
    mask &= x["adj_high"] > 0
    mask &= x["adj_low"] > 0
    mask &= x["adj_close"] > 0
    mask &= x["adj_high"] >= x[["adj_open", "adj_close", "adj_low"]].max(axis=1)
    mask &= x["adj_low"] <= x[["adj_open", "adj_close", "adj_high"]].min(axis=1)

    mask &= x["volume"] >= 0
    mask &= x["turnover"] >= 0

    invalid_count = int((~mask).sum())
    if invalid_count:
        message = f"invalid price rows: {invalid_count}/{n_before}"
        if strict:
            invalid = x.loc[~mask].head(10)
            raise ValueError(f"{message}\n{invalid.to_string(index=False)}")
        log.warning(message)
        x = x.loc[mask].copy()

    duplicate_mask = x.duplicated(subset=["code", "date"], keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        message = f"duplicate code-date rows: {duplicate_count}"

        if strict:
            duplicate = x.loc[duplicate_mask].head(10)
            raise ValueError(f"{message}\n{duplicate.to_string(index=False)}")

        log.warning(message)
        x = x.drop_duplicates(
            subset=["code", "date"],
            keep="last",
        )

    x["date"] = x["date"].dt.strftime("%Y-%m-%d")

    return x[REQUIRED_PRICE_COLS].sort_values(["code", "date"]).reset_index(drop=True)
