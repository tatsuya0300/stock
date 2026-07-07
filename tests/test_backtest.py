"""Backtester の約定・コストロジックの回帰テスト。"""

import numpy as np
import pandas as pd

from jp_signal.backtest import Backtester


def _prices():
    """2銘柄・数日分の固定価格。ADV検証のため turnover を明示。"""
    rows = [
        # code, date, open, high, low, close, volume, turnover
        ("A", "2024-01-04", 100, 105, 99, 104, 1000, 100000),
        ("A", "2024-01-05", 104, 110, 103, 108, 1000, 108000),
        ("A", "2024-01-09", 108, 112, 107, 110, 1000, 110000),
    ]
    df = pd.DataFrame(
        rows,
        columns=["code", "date", "open", "high", "low", "close", "volume", "turnover"],
    )
    return df


def test_buy_market_open_roundtrip_impact():
    """買い→翌日決済で往復インパクトが両サイドに乗ることを確認。"""
    px = _prices()
    sig = pd.DataFrame(
        [{
            "code": "A", "date": "2024-01-05", "side": "BUY",
            "qty": 100, "order_type": "MKT_OPEN",
            "limit_price": np.nan, "holding_days": 1,
        }]
    )
    bt = Backtester(impact_k_bp=30.0, commission_bp=0.0, half_spread_bp=0.0)
    res = bt.run(sig, px)
    row = res.iloc[0]
    assert row["status"] == "FILLED"
    # エントリーは 2024-01-05 の open=104 に買い方向スリッページで上振れ
    assert row["entry"] > 104
    # エグジットは 2024-01-09 の close=110 に売り方向スリッページで下振れ
    assert row["exit"] < 110


def test_short_skipped_without_shortability():
    """shortability 未提供時、SELL は保守的にスキップされる。"""
    px = _prices()
    sig = pd.DataFrame(
        [{
            "code": "A", "date": "2024-01-05", "side": "SELL",
            "qty": 100, "order_type": "MKT_OPEN",
            "limit_price": np.nan, "holding_days": 1,
        }]
    )
    bt = Backtester()
    res = bt.run(sig, px, shortability=None)
    assert res.iloc[0]["status"] == "SKIP_NOT_SHORTABLE"


def test_limit_buy_no_fill_on_equal_low():
    """安値==指値は未約定（同値未約定の仕様）。"""
    px = _prices()  # 2024-01-05 の low=103
    sig = pd.DataFrame(
        [{
            "code": "A", "date": "2024-01-05", "side": "BUY",
            "qty": 100, "order_type": "LIMIT",
            "limit_price": 103, "holding_days": 1,
        }]
    )
    bt = Backtester()
    res = bt.run(sig, px)
    assert res.iloc[0]["status"] == "NO_FILL"


def test_no_impact_when_adv_zero():
    """初日エントリーは前日 ADV が無いためインパクト0（約定価格=基準価格）。"""
    px = _prices()
    sig = pd.DataFrame(
        [{
            "code": "A", "date": "2024-01-04", "side": "BUY",
            "qty": 100, "order_type": "MKT_OPEN",
            "limit_price": np.nan, "holding_days": 1,
        }]
    )
    bt = Backtester(impact_k_bp=30.0)
    res = bt.run(sig, px)
    row = res.iloc[0]
    assert row["status"] == "FILLED"
    # 前日データが無い→ADV=0→インパクト0→open=100 で約定
    assert row["entry"] == 100.0
