"""storage tests."""

import pandas as pd

from jp_signal.storage import Storage


def test_storage_upsert_and_load_prices(tmp_path):
    db = tmp_path / "test.sqlite"
    st = Storage(str(db))

    try:
        df = pd.DataFrame(
            [
                {
                    "code": "7203",
                    "date": "2024-01-04",
                    "open": 100,
                    "high": 110,
                    "low": 90,
                    "close": 105,
                    "adj_open": 100,
                    "adj_high": 110,
                    "adj_low": 90,
                    "adj_close": 105,
                    "volume": 1000,
                    "turnover": 105000,
                }
            ]
        )

        st.upsert_prices(df)
        out = st.load_prices(["7203"], "2024-01-01", "2024-01-31")

        assert len(out) == 1
        assert out.iloc[0]["code"] == "7203"
        assert out.iloc[0]["adj_close"] == 105
        assert out.iloc[0]["close"] == 105

    finally:
        st.close()


def test_storage_upsert_prices_old_schema_input(tmp_path):
    """adj_* がない旧形式入力でも保存できる"""
    db = tmp_path / "test.sqlite"
    st = Storage(str(db))

    try:
        df = pd.DataFrame(
            [
                {
                    "code": "7203",
                    "date": "2024-01-04",
                    "open": 100,
                    "high": 110,
                    "low": 90,
                    "close": 105,
                    "volume": 1000,
                    "turnover": 105000,
                }
            ]
        )

        st.upsert_prices(df)
        out = st.load_prices(["7203"], "2024-01-01", "2024-01-31")

        assert len(out) == 1
        assert out.iloc[0]["adj_open"] == 100
        assert out.iloc[0]["adj_close"] == 105

    finally:
        st.close()


def test_append_signals_and_orders(tmp_path):
    db = tmp_path / "test.sqlite"
    st = Storage(str(db))

    try:
        sig = pd.DataFrame(
            [
                {
                    "code": "7203",
                    "side": "BUY",
                    "score": 1.23,
                }
            ]
        )

        st.append_signals(
            run_id="run1",
            signals=sig,
            signal_asof_date="2024-01-04",
            model_name="TestModel",
        )

        orders = pd.DataFrame(
            [
                {
                    "order_date": "2024-01-05",
                    "signal_asof_date": "2024-01-04",
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

        st.append_orders("run1", orders)

        n_sig = st.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        n_ord = st.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

        assert n_sig == 1
        assert n_ord == 1

    finally:
        st.close()
