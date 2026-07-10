"""Backtester short policy regression tests."""

from __future__ import annotations

import pandas as pd

from jp_signal.backtest import Backtester


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=30)
    rows = []
    for i, d in enumerate(dates):
        base = 1000.0 + i
        rows.append(
            {
                "code": "7203",
                "date": d.strftime("%Y-%m-%d"),
                "open": base,
                "high": base + 10,
                "low": base - 10,
                "close": base + 1,
                "adj_open": base,
                "adj_high": base + 10,
                "adj_low": base - 10,
                "adj_close": base + 1,
                "volume": 1000000,
                "turnover": 1_000_000_000,
            }
        )
    return pd.DataFrame(rows)


def _sell_signal() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2026-01-29",
                "side": "SELL",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "limit_price": None,
                "holding_days": 1,
            }
        ]
    )


def test_sell_without_shortability_is_skipped_by_default() -> None:
    bt = Backtester(require_confirmed_shortability=True)
    result = bt.run(_sell_signal(), _prices(), shortability=None)

    assert len(result) == 1
    assert result.iloc[0]["status"] == "SKIP_NOT_SHORTABLE"


def test_sell_without_shortability_can_be_allowed_for_research() -> None:
    bt = Backtester(require_confirmed_shortability=False)
    result = bt.run(_sell_signal(), _prices(), shortability=None)

    assert len(result) == 1
    assert result.iloc[0]["status"] == "FILLED"
