"""バックテスト指標のテスト。"""

import pandas as pd
import pytest

from jp_signal.metrics import summarize_backtest


def test_pnl_is_recorded_on_exit_date() -> None:
    result = pd.DataFrame(
        [
            {
                "date": "2026-07-06",
                "entry_date": "2026-07-06",
                "exit_date": "2026-07-07",
                "status": "FILLED",
                "pnl": 1_000.0,
            }
        ]
    )

    summary = summarize_backtest(
        result,
        initial_capital=1_000_000,
        trading_dates=[
            "2026-07-06",
            "2026-07-07",
        ],
    )

    daily_pnl = summary["daily_pnl"]

    assert daily_pnl.loc[pd.Timestamp("2026-07-06")] == 0
    assert daily_pnl.loc[pd.Timestamp("2026-07-07")] == 1_000


def test_initial_loss_is_included_in_drawdown() -> None:
    result = pd.DataFrame(
        [
            {
                "exit_date": "2026-07-06",
                "status": "FILLED",
                "pnl": -10_000.0,
            }
        ]
    )

    summary = summarize_backtest(
        result,
        initial_capital=1_000_000,
        trading_dates=["2026-07-06"],
    )

    assert summary["max_drawdown_yen"] == pytest.approx(-10_000)
    assert summary["max_drawdown_pct"] == pytest.approx(-0.01)
    assert summary["total_return"] == pytest.approx(-0.01)


def test_sharpe_uses_returns_not_yen_pnl() -> None:
    result = pd.DataFrame(
        [
            {
                "exit_date": "2026-07-06",
                "status": "FILLED",
                "pnl": 10_000.0,
            },
            {
                "exit_date": "2026-07-07",
                "status": "FILLED",
                "pnl": -5_000.0,
            },
        ]
    )

    summary = summarize_backtest(
        result,
        initial_capital=1_000_000,
        trading_dates=[
            "2026-07-06",
            "2026-07-07",
        ],
    )

    assert "sharpe" in summary
    assert "daily_returns" in summary
    assert len(summary["daily_returns"]) == 2


def test_non_positive_initial_capital_is_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_backtest(
            pd.DataFrame(),
            initial_capital=0,
        )
