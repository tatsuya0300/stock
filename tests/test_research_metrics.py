from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jp_signal.research_metrics import (
    benchmark_statistics,
    drawdown_series_from_returns,
    estimate_trade_turnover,
    holm_adjust,
    ledger_returns,
    summarize_capacity,
    summarize_research_performance,
)


def test_ledger_returns_includes_first_day():
    ledger = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "nav": 990_000,
            },
            {
                "date": "2024-01-03",
                "nav": 999_900,
            },
        ]
    )

    returns = ledger_returns(
        ledger,
        initial_capital=1_000_000,
    )

    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(-0.01)
    assert returns.iloc[1] == pytest.approx(0.01)


def test_performance_positive_strategy():
    ledger = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "nav": 1_010_000,
                "gross_exposure": 500_000,
                "net_exposure": 10_000,
            },
            {
                "date": "2024-01-03",
                "nav": 1_020_000,
                "gross_exposure": 400_000,
                "net_exposure": -10_000,
            },
            {
                "date": "2024-01-04",
                "nav": 1_030_000,
                "gross_exposure": 0,
                "net_exposure": 0,
            },
        ]
    )

    result = summarize_research_performance(
        ledger,
        initial_capital=1_000_000,
    )

    assert result["total_return"] == pytest.approx(0.03)
    assert result["annualized_return"] > 0
    assert result["average_gross_exposure"] == pytest.approx(300_000)


def test_benchmark_statistics_beta():
    dates = pd.date_range(
        "2024-01-01",
        periods=5,
        freq="B",
    )

    benchmark = pd.Series(
        [-0.02, -0.01, 0.0, 0.01, 0.02],
        index=dates,
    )
    strategy = 0.001 + 2.0 * benchmark

    result = benchmark_statistics(
        strategy,
        benchmark,
    )

    assert result["benchmark_beta"] == pytest.approx(2.0)
    assert result["benchmark_alpha_annualized"] == pytest.approx(0.001 * 252)


def test_turnover_counts_entry_and_exit():
    trades = pd.DataFrame(
        [
            {
                "entry_date": "2024-01-02",
                "exit_date": "2024-01-03",
                "entry": 100.0,
                "exit": 110.0,
                "qty": 100,
            }
        ]
    )
    ledger = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "nav": 1_000_000,
            },
            {
                "date": "2024-01-03",
                "nav": 1_001_000,
            },
        ]
    )

    result = estimate_trade_turnover(
        trades,
        ledger,
        initial_capital=1_000_000,
    )

    assert result.iloc[0]["traded_notional"] == pytest.approx(10_000)
    assert result.iloc[1]["traded_notional"] == pytest.approx(11_000)


def test_capacity_summary():
    trades = pd.DataFrame(
        [
            {
                "entry": 100.0,
                "exit": 101.0,
                "qty": 100,
                "entry_adv": 1_000_000,
                "exit_adv": 1_000_000,
            }
        ]
    )

    result = summarize_capacity(trades)

    assert result["execution_count"] == 2
    assert result["participation_mean"] == pytest.approx(0.01005)


def test_holm_adjust():
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert np.allclose(
        adjusted,
        [0.03, 0.06, 0.06],
    )


def test_holm_rejects_invalid_p_values():
    with pytest.raises(ValueError):
        holm_adjust([0.1, 1.2])


def test_drawdown_includes_first_day_loss():
    returns = pd.Series(
        [-0.10, 0.05],
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
            ]
        ),
    )

    drawdown = drawdown_series_from_returns(returns)

    assert drawdown.iloc[0] == pytest.approx(-0.10)
    assert drawdown.min() == pytest.approx(-0.10)


def test_research_performance_includes_first_day_drawdown():
    ledger = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "nav": 900_000,
            },
            {
                "date": "2024-01-03",
                "nav": 945_000,
            },
        ]
    )

    result = summarize_research_performance(
        ledger,
        initial_capital=1_000_000,
    )

    assert result["max_drawdown"] == pytest.approx(-0.10)


def test_benchmark_statistics_zero_variance():
    """benchmark varianceがゼロの場合にゼロ割が発生しないことを確認する。"""
    dates = pd.date_range(
        "2024-01-01",
        periods=5,
        freq="B",
    )

    benchmark = pd.Series(
        [0.01] * 5,
        index=dates,
    )
    strategy = pd.Series(
        [0.02] * 5,
        index=dates,
    )

    result = benchmark_statistics(
        strategy,
        benchmark,
    )

    assert result["benchmark_observations"] == 5
    assert result["benchmark_beta"] == 0.0
