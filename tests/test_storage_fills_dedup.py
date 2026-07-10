from pathlib import Path

import pandas as pd

from jp_signal.storage import Storage


def test_import_fills_csv_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    csv_path = tmp_path / "fills.csv"

    df = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-10",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "price": 1000.0,
            },
            {
                "trade_date": "2024-01-10",
                "code": "6758",
                "side": "SELL",
                "qty": 100,
                "price": 2000.0,
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    with Storage(str(db_path)) as st:
        n1 = st.import_fills_csv(csv_path)
        n2 = st.import_fills_csv(csv_path)
        fills = st.load_fills(trade_date="2024-01-10")

    assert n1 == 2
    assert n2 == 0
    assert len(fills) == 2
