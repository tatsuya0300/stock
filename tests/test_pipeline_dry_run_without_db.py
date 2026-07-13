"""PR-2: dry-run without existing database tests."""
from datetime import date

import pandas as pd

from jp_signal import pipeline


class FakeDataSource:
    def fetch_daily(self, codes, start, end):
        dates = pd.date_range(end=end, periods=30, freq="B")
        rows = []
        for code in codes:
            for i, day in enumerate(dates):
                price = 100.0 + i
                rows.append({
                    "code": code,
                    "date": str(day.date()),
                    "open": price, "high": price + 1, "low": price - 1, "close": price,
                    "adj_open": price, "adj_high": price + 1, "adj_low": price - 1, "adj_close": price,
                    "volume": 100_000, "turnover": price * 100_000,
                })
        return pd.DataFrame(rows)


class FakeNotifier:
    def send(self, title, body):
        return None


def test_dry_run_does_not_require_existing_database(tmp_path, monkeypatch):
    """Dry-run must work without an existing database file."""
    universe_path = tmp_path / "universe.csv"
    pd.DataFrame({"code": ["1111", "2222"], "name": ["A", "B"]}).to_csv(universe_path, index=False)

    monkeypatch.setattr(pipeline, "make_datasource", lambda cfg: FakeDataSource())
    monkeypatch.setattr(pipeline, "make_notifier", lambda cfg: FakeNotifier())

    cfg = {
        "data": {
            "source": "yfinance",
            "db_path": str(tmp_path / "missing.db"),
            "allow_approximate_turnover": True,
        },
        "universe": {"file": str(universe_path)},
        "model": {"lookback": 5, "top_n": 1},
        "backtest": {"adv_window": 20, "min_adv_periods": 1, "impact_k_is_calibrated": False},
        "sizing": {
            "adv_ratio": 0.001, "adv_ratio_cap": 0.002, "adv_window": 20,
            "min_adv_periods": 1, "require_full_adv_history": False,
            "allow_single_day_turnover_fallback": True, "unit": 100,
        },
        "risk": {
            "max_orders_per_day": 10, "max_gross_exposure_yen": 100_000_000,
            "max_single_name_exposure_yen": 20_000_000, "max_long_exposure_yen": 50_000_000,
            "max_short_exposure_yen": 50_000_000, "max_net_exposure_yen": 50_000_000,
            "require_both_sides": False, "allow_short_without_confirmed_shortability": False,
        },
        "data_quality": {"hard_fail": False},
        "notify": {"channel": "console", "morning_time": "08:15"},
    }

    pipeline.morning_pipeline(date(2026, 7, 13), cfg, dry_run=True)
    assert not (tmp_path / "missing.db").exists()
