"""価格データ品質チェック（FR-QUALITY-01/02）。
スケルトン — Part 2 で完全実装。
"""

from __future__ import annotations

import pandas as pd

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


def validate_prices(
    df: pd.DataFrame, strict: bool = False
) -> pd.DataFrame:
    """Validate price data rows.

    Args:
        df: Price DataFrame.
        strict: If True, raises ValueError on invalid rows.
                If False, drops invalid rows silently.

    Returns:
        Filtered DataFrame with only valid rows.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_PRICE_COLS)

    # Check required columns exist
    missing = [c for c in ["code", "open", "high", "low"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    required = [c for c in REQUIRED_PRICE_COLS if c in df.columns]
    for c in ["code", "date", "open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    for c in ["adj_open", "adj_high", "adj_low", "adj_close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Validate high >= low
    valid_high_low: pd.Series | bool = True
    if "high" in df.columns and "low" in df.columns:
        valid_high_low = df["high"] >= df["low"]
    if "adj_high" in df.columns and "adj_low" in df.columns:
        if isinstance(valid_high_low, bool):
            valid_high_low = df["adj_high"] >= df["adj_low"]
        else:
            valid_high_low = valid_high_low & (df["adj_high"] >= df["adj_low"])

    if isinstance(valid_high_low, pd.Series):
        invalid = ~valid_high_low
        if invalid.any():
            if strict:
                raise ValueError(
                    f"Found {invalid.sum()} rows with high < low"
                )
            df = df[~invalid]
    elif strict and not valid_high_low:
        raise ValueError("Found rows with high < low (detected at bool level)")

    return df[required].copy()
