"""バックテスト結果の集計・評価指標。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252.0


def summarize_backtest(
    result: pd.DataFrame,
    *,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    trading_days_per_year: float = _TRADING_DAYS_PER_YEAR,
    trading_dates: Sequence[str] | None = None,
) -> dict:
    """バックテスト結果を集計する。

    Sharpe:
      日次PnLではなく、前日NAVに対する日次リターンから計算する。

    PnL計上日:
      exit_dateが存在すればexit日に計上する。
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    empty_summary = {
        "n_signals": 0,
        "n_filled": 0,
        "fill_rate": 0.0,
        "total_pnl": 0.0,
        "total_return": 0.0,
        "win_rate": 0.0,
        "daily_return_mean": 0.0,
        "daily_return_std": 0.0,
        "sharpe": 0.0,
        "max_drawdown_yen": 0.0,
        "max_drawdown_pct": 0.0,
        "status_counts": {},
    }

    if result is None or result.empty:
        return empty_summary

    n_signals = len(result)

    status_counts = result["status"].value_counts().to_dict() if "status" in result.columns else {}

    if "status" in result.columns:
        filled = result[result["status"] == "FILLED"].copy()
    else:
        filled = result.copy()

    n_filled = len(filled)
    fill_rate = n_filled / n_signals if n_signals else 0.0

    if filled.empty or "pnl" not in filled.columns:
        return {
            **empty_summary,
            "n_signals": n_signals,
            "n_filled": n_filled,
            "fill_rate": fill_rate,
            "status_counts": status_counts,
        }

    filled["pnl"] = pd.to_numeric(
        filled["pnl"],
        errors="coerce",
    )

    if filled["pnl"].isna().any():
        raise ValueError("filled trades contain NaN pnl")

    pnl_date_col = "exit_date" if "exit_date" in filled.columns else "date"

    filled[pnl_date_col] = pd.to_datetime(
        filled[pnl_date_col],
        errors="raise",
    )

    daily_pnl = filled.groupby(pnl_date_col, sort=True)["pnl"].sum().astype(float)

    if trading_dates is not None:
        calendar_index = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).sort_values()

        daily_pnl = daily_pnl.reindex(
            calendar_index,
            fill_value=0.0,
        )

    total_pnl = float(daily_pnl.sum())
    win_rate = float((filled["pnl"] > 0).mean())

    # 初期資産を先頭に追加
    initial_date = daily_pnl.index.min() - pd.Timedelta(days=1)

    equity_curve = pd.concat(
        [
            pd.Series(
                [initial_capital],
                index=[initial_date],
                dtype=float,
            ),
            initial_capital + daily_pnl.cumsum(),
        ]
    )

    prior_nav = equity_curve.shift(1)
    daily_returns = (equity_curve / prior_nav - 1.0).iloc[1:]

    if (prior_nav.iloc[1:] <= 0).any():
        raise ValueError("NAV became non-positive; return-based metrics are undefined")

    daily_return_mean = float(daily_returns.mean()) if len(daily_returns) else 0.0

    daily_return_std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0

    if daily_return_std > 0:
        sharpe = float(
            (daily_return_mean - daily_rf) / daily_return_std * np.sqrt(trading_days_per_year)
        )
    else:
        sharpe = 0.0

    running_peak = equity_curve.cummax()
    drawdown_yen = equity_curve - running_peak
    drawdown_pct = equity_curve / running_peak - 1.0

    total_return = float(equity_curve.iloc[-1] / initial_capital - 1.0)

    return {
        "n_signals": n_signals,
        "n_filled": n_filled,
        "fill_rate": fill_rate,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "win_rate": win_rate,
        "daily_return_mean": daily_return_mean,
        "daily_return_std": daily_return_std,
        "sharpe": sharpe,
        "max_drawdown_yen": float(drawdown_yen.min()),
        "max_drawdown_pct": float(drawdown_pct.min()),
        "status_counts": status_counts,
        "daily_pnl": daily_pnl,
        "daily_returns": daily_returns,
        "equity_curve": equity_curve,
    }
