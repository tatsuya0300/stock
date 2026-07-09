"""model tests."""

import pandas as pd

from jp_signal.model import MeanReversionRule


def test_mean_reversion_uses_adj_close():
    prices = pd.DataFrame(
        [
            # code A: raw closeは上昇、adj_closeは下落
            ("A", "2024-01-01", 100, 100),
            ("A", "2024-01-02", 110, 90),
            # code B: raw closeは横ばい、adj_closeは上昇
            ("B", "2024-01-01", 100, 100),
            ("B", "2024-01-02", 100, 120),
        ],
        columns=["code", "date", "close", "adj_close"],
    )

    model = MeanReversionRule(lookback=1, top_n=1)
    sig = model.generate(prices, as_of="2024-01-02")

    buy_codes = sig.loc[sig["side"] == "BUY", "code"].tolist()
    sell_codes = sig.loc[sig["side"] == "SELL", "code"].tolist()

    assert buy_codes == ["A"]
    assert sell_codes == ["B"]
