"""価格データ品質チェック（FR-QUALITY-01/02）。"""

from __future__ import annotations

import logging

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


def validate_prices(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate price data rows.

    - 必須列の存在確認
    - 数値列を coerce
    - open/high/low/close/adj_close が正の値であること
    - high >= low, adj_high >= adj_low
    - volume/turnover >= 0
    - 重複 (code, date) を除去

    Args:
        df: Price DataFrame.
        strict: True の場合、不正行を ValueError で報告。
                False の場合、警告ログを出して削除。

    Returns:
        Filtered DataFrame (REQUIRED_PRICE_COLS のみ)。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_PRICE_COLS)

    missing = [c for c in REQUIRED_PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df.copy()
    n0 = len(x)

    # 数値列を強制変換
    num_cols = [c for c in REQUIRED_PRICE_COLS if c not in ("code", "date")]
    for c in num_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x["code"] = x["code"].astype(str).str.strip()

    mask = x["code"].str.len() > 0
    mask &= x["date"].notna()

    for c in ["open", "high", "low", "close", "adj_close"]:
        mask &= x[c].notna() & (x[c] > 0)

    mask &= x["high"] >= x["low"]
    mask &= x["adj_high"] >= x["adj_low"]
    mask &= x["volume"].fillna(0) >= 0
    mask &= x["turnover"].fillna(0) >= 0

    invalid_n = int((~mask).sum())
    if invalid_n:
        msg = f"invalid price rows: {invalid_n}/{n0}"
        if strict:
            raise ValueError(msg)
        log.warning(msg)
        x = x.loc[mask]

    # 重複 (code, date) を除去
    dup = x.duplicated(subset=["code", "date"], keep="last")
    if dup.any():
        msg = f"duplicate code-date rows dropped: {int(dup.sum())}"
        if strict:
            raise ValueError(msg)
        log.warning(msg)
        x = x.loc[~dup]

    return x[REQUIRED_PRICE_COLS].reset_index(drop=True)
