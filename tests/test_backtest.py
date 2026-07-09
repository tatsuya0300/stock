"""Backtester の約定・売り可否・流動性処理テスト。"""

import numpy as np
import pandas as pd
import pytest

from jp_signal.backtest import Backtester


def _prices():
    rows = [
        ("A", "2024-01-04", 100, 105, 99, 104, 1000, 100000),
        ("A", "2024-01-05", 104, 110, 103, 108, 1000, 108000),
        ("A", "2024-01-09", 108, 112, 107, 110, 1000, 110000),
    ]
    return pd.DataFrame(
        rows,
        columns=["code", "date", "open", "high", "low", "close", "volume", "turnover"],
    )


def _buy_signal(date="2024-01-05", qty=100):
    return pd.DataFrame(
        [
            {
                "code": "A",
                "date": date,
                "side": "BUY",
                "qty": qty,
                "order_type": "MKT_OPEN",
                "limit_price": np.nan,
                "holding_days": 1,
            }
        ]
    )


def _sell_signal(date="2024-01-05", qty=100):
    return pd.DataFrame(
        [
            {
                "code": "A",
                "date": date,
                "side": "SELL",
                "qty": qty,
                "order_type": "MKT_OPEN",
                "limit_price": np.nan,
                "holding_days": 1,
            }
        ]
    )


def test_buy_market_open_roundtrip_impact():
    px = _prices()
    sig = _buy_signal()

    bt = Backtester(impact_k_bp=30.0, require_liquidity_data=True)
    res = bt.run(sig, px)
    row = res.iloc[0]

    assert row["status"] == "FILLED"
    assert row["entry"] > 104
    assert row["exit"] < 110


def test_short_skipped_without_shortability():
    px = _prices()
    sig = _sell_signal()

    bt = Backtester()
    res = bt.run(sig, px, shortability=None)

    assert res.iloc[0]["status"] == "SKIP_NOT_SHORTABLE"


def test_short_uses_latest_snapshot_before_order_date():
    px = _prices()
    sig = _sell_signal()

    short = pd.DataFrame(
        [
            ("A", "2024-01-04", 1, 0),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    bt = Backtester()
    res = bt.run(sig, px, shortability=short)

    assert res.iloc[0]["status"] == "FILLED"


def test_short_restricted_is_skipped():
    px = _prices()
    sig = _sell_signal()

    short = pd.DataFrame(
        [
            ("A", "2024-01-04", 1, 1),
        ],
        columns=["code", "date", "is_margin_lendable", "short_restricted"],
    )

    bt = Backtester()
    res = bt.run(sig, px, shortability=short)

    assert res.iloc[0]["status"] == "SKIP_NOT_SHORTABLE"


def test_limit_buy_no_fill_on_equal_low():
    px = _prices()
    sig = pd.DataFrame(
        [
            {
                "code": "A",
                "date": "2024-01-05",
                "side": "BUY",
                "qty": 100,
                "order_type": "LIMIT",
                "limit_price": 103,
                "holding_days": 1,
            }
        ]
    )

    bt = Backtester()
    res = bt.run(sig, px)

    assert res.iloc[0]["status"] == "NO_FILL"


def test_no_liquidity_data_skips_when_required():
    px = _prices()
    sig = _buy_signal(date="2024-01-04")

    bt = Backtester(require_liquidity_data=True)
    res = bt.run(sig, px)

    assert res.iloc[0]["status"] == "NO_ENTRY_LIQUIDITY_DATA"


def test_no_impact_when_liquidity_check_disabled():
    px = _prices()
    sig = _buy_signal(date="2024-01-04")

    bt = Backtester(impact_k_bp=30.0, require_liquidity_data=False)
    res = bt.run(sig, px)

    assert res.iloc[0]["status"] == "FILLED"
    assert res.iloc[0]["entry"] == 100.0


def test_invalid_side_returns_status():
    px = _prices()
    sig = _buy_signal()
    sig.loc[0, "side"] = "HOLD"

    bt = Backtester()
    res = bt.run(sig, px)

    assert res.iloc[0]["status"] == "INVALID_SIDE"


def test_invalid_qty_returns_status():
    px = _prices()
    sig = _buy_signal(qty=0)

    bt = Backtester()
    res = bt.run(sig, px)

    assert res.iloc[0]["status"] == "INVALID_QTY"


def test_missing_required_signal_column_raises():
    px = _prices()
    sig = _buy_signal().drop(columns=["qty"])

    bt = Backtester()

    with pytest.raises(ValueError):
        bt.run(sig, px)
