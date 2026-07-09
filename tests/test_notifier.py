"""notifier tests."""

import pandas as pd

from jp_signal.notifier import _split_text, format_orders


def test_split_text_short():
    assert _split_text("abc", 10) == ["abc"]


def test_split_text_long():
    text = "\n".join([f"line{i}" for i in range(20)])
    chunks = _split_text(text, 30)

    assert len(chunks) > 1
    assert all(len(c) <= 30 for c in chunks)


def test_format_orders_empty():
    body = format_orders(pd.DataFrame())

    assert "注文なし" in body


def test_format_orders():
    orders = pd.DataFrame(
        [
            {
                "signal_asof_date": "2024-01-04",
                "order_date": "2024-01-05",
                "code": "7203",
                "name": "Toyota",
                "side": "BUY",
                "order_type": "MKT_OPEN",
                "qty": 100,
                "ref_price": 1000,
                "value_yen": 100000,
                "shortable": True,
                "warn": "",
            }
        ]
    )

    body = format_orders(orders)

    assert "7203" in body
    assert "Toyota" in body
    assert "MKT_OPEN" in body
