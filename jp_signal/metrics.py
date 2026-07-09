"""バックテスト結果の集計・評価指標（FR-BT-METRICS）。

提供指標:
  - 約定率 / 勝率 / 合計PnL
  - 日次PnLの平均・標準偏差
  - Sharpe-like 比率（年率換算）
  - Max Drawdown（PnL ベース）
  - status 内訳
  - 日次PnL系列・資産曲線
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 簡易 Sharpe の年率換算用（252営業日想定）
_TRADING_DAYS_PER_YEAR = 252.0


def summarize_backtest(
    result: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days_per_year: float = _TRADING_DAYS_PER_YEAR,
) -> dict:
    """Backtester.run の結果からサマリを返す。

    PnL は円（名目）前提。リターン正規化は initial_capital が無いため
    日次 PnL ベースの簡易 Sharpe のみ。
    """
    if result is None or result.empty:
        return {
            "n_signals": 0,
            "n_filled": 0,
            "fill_rate": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "daily_pnl_mean": 0.0,
            "daily_pnl_std": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown_pnl": 0.0,
            "status_counts": {},
        }

    n_signals = len(result)
    status_counts = result["status"].value_counts().to_dict() if "status" in result else {}
    filled = result[result["status"] == "FILLED"].copy() if "status" in result else result
    n_filled = len(filled)
    fill_rate = n_filled / max(n_signals, 1)

    if filled.empty or "pnl" not in filled.columns:
        return {
            "n_signals": n_signals,
            "n_filled": n_filled,
            "fill_rate": fill_rate,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "daily_pnl_mean": 0.0,
            "daily_pnl_std": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown_pnl": 0.0,
            "status_counts": status_counts,
        }

    total_pnl = float(filled["pnl"].sum())
    win_rate = float((filled["pnl"] > 0).mean())

    # 約定日ベースの日次 PnL（entry 日）
    daily = filled.groupby("date", sort=True)["pnl"].sum()
    daily_mean = float(daily.mean()) if len(daily) else 0.0
    daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0

    if daily_std > 0:
        excess = daily_mean - (risk_free_rate / trading_days_per_year)
        sharpe_like = float(excess / daily_std * np.sqrt(trading_days_per_year))
    else:
        sharpe_like = 0.0

    equity = daily.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    return {
        "n_signals": n_signals,
        "n_filled": n_filled,
        "fill_rate": fill_rate,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "daily_pnl_mean": daily_mean,
        "daily_pnl_std": daily_std,
        "sharpe_like": sharpe_like,
        "max_drawdown_pnl": max_dd,
        "status_counts": status_counts,
        "daily_pnl": daily,
        "equity_curve": equity,
    }
