"""ADV計算ユーティリティ。

live / backtest で同じADV計算を使う。
look-aheadを避けるため、原則として as_of 当日のデータは使わない。
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def rolling_adv_before(
    prices: pd.DataFrame,
    as_of: date | str | pd.Timestamp,
    *,
    window: int,
    min_periods: int,
    code_col: str = "code",
    date_col: str = "date",
    turnover_col: str = "turnover",
    strictly_before: bool = True,
) -> pd.Series:
    """as_of以前のrolling ADVを銘柄別に返す。

    Args:
        prices:
            code/date/turnoverを含むDataFrame。
        as_of:
            基準日。
        window:
            ADV計算窓。
        min_periods:
            最低必要営業日数。
        strictly_before:
            Trueなら date < as_of （as_of当日を使わない）。
            Falseなら date <= as_of （as_of当日も使う）。BT決済時のexit側ADVで使用。
        code_col:
            コード列名。
        date_col:
            日付列名。
        turnover_col:
            売買代金列名。

    Returns:
        code → ADV の Series。履歴不足なら NaN。
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float)

    if window < 1:
        raise ValueError("window must be >= 1")
    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")
    if min_periods > window:
        raise ValueError("min_periods must be <= window")

    as_of_ts = pd.Timestamp(as_of)

    x = prices.copy()
    x[date_col] = pd.to_datetime(x[date_col])
    x[turnover_col] = pd.to_numeric(x[turnover_col], errors="coerce")

    if strictly_before:
        x = x[x[date_col] < as_of_ts]
    else:
        x = x[x[date_col] <= as_of_ts]

    if x.empty:
        return pd.Series(dtype=float)

    x = x.sort_values([code_col, date_col])

    def _mean_last_window(s: pd.Series) -> float:
        # tail を先に取ってから NaN を除外する。
        # こうすることで直近 window 行中の NaN が min_periods の対象になる。
        tail = s.tail(window)
        valid = tail.dropna()
        if len(valid) < min_periods:
            return float("nan")
        return float(valid.mean())

    return x.groupby(code_col)[turnover_col].apply(_mean_last_window).astype(float)


def stock_adv_before(
    prices: pd.DataFrame,
    as_of: date | str | pd.Timestamp,
    code: int | str,
    *,
    window: int,
    min_periods: int,
    strictly_before: bool = True,
) -> float:
    """単一銘柄のADVを返す。履歴不足ならNaN。"""
    adv = rolling_adv_before(
        prices,
        as_of,
        window=window,
        min_periods=min_periods,
        strictly_before=strictly_before,
    )
    return float(adv.get(str(code), float("nan")))
