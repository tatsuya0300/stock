"""MeanReversionRule の look-ahead 回避テスト。"""

import pandas as pd

from jp_signal.model import MeanReversionRule


def test_generate_uses_previous_business_day_prices_only():
    """2024-01-09 は火曜。前営業日は 2024-01-08。
    当日(2024-01-09)の終値を使ってはいけない。
    """
    prices = pd.DataFrame(
        [
            ("A", "2024-01-04", 100.0),
            ("A", "2024-01-05", 90.0),
            ("A", "2024-01-09", 50.0),  # as_of 当日。使ってはいけない
            ("B", "2024-01-04", 100.0),
            ("B", "2024-01-05", 110.0),
            ("B", "2024-01-09", 200.0),
        ],
        columns=["code", "date", "adj_close"],
    )
    # 最低限必要な列を埋める
    for c in ["open", "high", "low", "close"]:
        prices[c] = prices["adj_close"]
    prices["volume"] = 1000
    prices["turnover"] = 100000

    model = MeanReversionRule(lookback=1, top_n=1)
    sig = model.generate(prices, as_of="2024-01-09")

    # A は下落 → BUY、B は上昇 → SELL が期待
    assert not sig.empty
    sides = dict(zip(sig["code"], sig["side"], strict=False))
    assert sides.get("A") == "BUY"
    assert sides.get("B") == "SELL"


def test_single_symbol_does_not_generate_overlapping_sides() -> None:
    prices = pd.DataFrame(
        [
            ("A", "2026-07-01", 100.0),
            ("A", "2026-07-02", 99.0),
            ("A", "2026-07-03", 98.0),
            ("A", "2026-07-06", 97.0),
            ("A", "2026-07-07", 96.0),
            ("A", "2026-07-08", 95.0),
        ],
        columns=["code", "date", "adj_close"],
    )
    prices["close"] = prices["adj_close"]
    for c in ["open", "high", "low"]:
        prices[c] = prices["adj_close"]
    prices["volume"] = 1000
    prices["turnover"] = 100000

    model = MeanReversionRule(
        lookback=5,
        top_n=1,
    )

    result = model.generate(
        prices,
        as_of="2026-07-09",
    )

    assert result.empty
