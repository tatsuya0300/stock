"""Portfolio metrics computation from daily NAV ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_portfolio_ledger(
    ledger: pd.DataFrame,
    *,
    initial_capital: float,
    risk_free_rate: float = 0.0,
) -> dict:
    """日次NAV ledgerから指標を計算する。"""
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0: {initial_capital}")

    if ledger is None or ledger.empty:
        return {"error": "ledger is empty"}

    nav = pd.to_numeric(ledger["nav"], errors="coerce").dropna()
    if len(nav) < 2:
        return {"error": f"ledger too short: {len(nav)} rows"}

    daily_returns = nav.pct_change().dropna()

    mean_return = float(daily_returns.mean())
    std_return = float(daily_returns.std())

    trading_days_per_year = 252.0
    annual_factor = (
        np.sqrt(trading_days_per_year) if std_return > 0 else 0.0
    )

    daily_rf = (
        (1.0 + risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0
    )

    if std_return > 0:
        sharpe = float(
            (mean_return - daily_rf) / std_return * annual_factor
        )
    else:
        sharpe = 0.0

    first_idx = nav.index.min()
    if isinstance(first_idx, pd.Timestamp):
        eq_index = [first_idx - pd.Timedelta(days=1)]
    else:
        eq_index = [first_idx - 1]
    equity_curve = pd.concat(
        [
            pd.Series(
                [initial_capital],
                index=eq_index,
                dtype=float,
            ),
            nav,
        ]
    )

    running_peak = equity_curve.cummax()
    drawdown_yen = equity_curve - running_peak
    drawdown_pct = equity_curve / running_peak - 1.0

    if "gross_exposure" in ledger.columns:
        gross = pd.to_numeric(ledger["gross_exposure"], errors="coerce").fillna(0.0)
    else:
        gross = pd.Series(0.0, index=ledger.index)

    if "net_exposure" in ledger.columns:
        net = pd.to_numeric(ledger["net_exposure"], errors="coerce").fillna(0.0)
    else:
        net = pd.Series(0.0, index=ledger.index)

    final_nav = float(nav.iloc[-1])

    return {
        "initial_capital": initial_capital,
        "final_nav": final_nav,
        "total_pnl": final_nav - initial_capital,
        "total_return": final_nav / initial_capital - 1.0,
        "daily_return_mean": mean_return,
        "daily_return_std": std_return,
        "sharpe": sharpe,
        "max_drawdown_yen": float(drawdown_yen.min()),
        "max_drawdown_pct": float(drawdown_pct.min()),
        "max_gross_exposure": float(gross.max()),
        "average_gross_exposure": float(gross.mean()),
        "max_abs_net_exposure": float(net.abs().max()),
        "daily_returns": daily_returns,
        "equity_curve": equity_curve,
    }
