"""Portfolio metrics tests."""

from __future__ import annotations

import pandas as pd

from jp_signal.portfolio_metrics import summarize_portfolio_ledger


def test_summarize_empty_ledger():
    result = summarize_portfolio_ledger(
        pd.DataFrame(),
        initial_capital=1_000_000,
    )
    assert "error" in result


def test_summarize_single_row_ledger():
    ledger = pd.DataFrame(
        [{"nav": 1_000_000, "gross_exposure": 0.0}]
    )
    result = summarize_portfolio_ledger(
        ledger,
        initial_capital=1_000_000,
    )
    assert "error" in result
    assert "too short" in result["error"]


def test_summarize_positive_return():
    ledger = pd.DataFrame(
        [
            {"nav": 1_000_000, "gross_exposure": 500_000},
            {"nav": 1_010_000, "gross_exposure": 500_000},
            {"nav": 1_020_000, "gross_exposure": 500_000},
        ]
    )
    result = summarize_portfolio_ledger(
        ledger,
        initial_capital=1_000_000,
    )

    assert result["initial_capital"] == 1_000_000
    assert result["final_nav"] == 1_020_000
    assert result["total_pnl"] == 20_000
    assert abs(result["total_return"] - 0.02) < 0.001
    assert result["sharpe"] > 0
    assert result["max_drawdown_pct"] >= -0.01
    assert result["max_gross_exposure"] == 500_000
