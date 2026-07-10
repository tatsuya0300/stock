"""Research-grade portfolio evaluation utilities.

評価対象:
- annualized return / volatility
- Sharpe / Sortino / Calmar
- maximum drawdown
- benchmark alpha / beta / correlation
- turnover
- ADV participation / capacity
- Holm multiple-testing correction
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252.0


def ledger_returns(
    ledger: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.Series:
    """日次ledgerから、初日を含む日次リターンを返す。"""
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be positive: {initial_capital}")

    if ledger is None or ledger.empty:
        return pd.Series(dtype=float, name="strategy_return")

    required = {"date", "nav"}
    missing = required - set(ledger.columns)

    if missing:
        raise ValueError(f"ledger missing columns: {sorted(missing)}")

    frame = ledger.copy()
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    )
    frame["nav"] = pd.to_numeric(
        frame["nav"],
        errors="raise",
    )

    frame = frame.sort_values("date").drop_duplicates("date", keep="last")

    if (frame["nav"] <= 0).any():
        raise ValueError("ledger contains non-positive NAV")

    nav = frame["nav"].to_numpy(dtype=float)
    previous = np.empty(len(nav), dtype=float)
    previous[0] = float(initial_capital)
    previous[1:] = nav[:-1]

    returns = nav / previous - 1.0

    return pd.Series(
        returns,
        index=frame["date"],
        name="strategy_return",
    )


def drawdown_series_from_returns(
    returns: pd.Series,
) -> pd.Series:
    """リターン系列からdrawdown系列を作る。"""
    clean = pd.to_numeric(
        returns,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return pd.Series(dtype=float, name="drawdown")

    equity = (1.0 + clean).cumprod()
    peak = equity.cummax()

    drawdown = equity / peak - 1.0
    drawdown.name = "drawdown"

    return drawdown


def _annualized_return(
    returns: pd.Series,
    trading_days_per_year: float,
) -> float:
    if returns.empty:
        return 0.0

    total_growth = float((1.0 + returns).prod())

    if total_growth <= 0:
        return -1.0

    n = float(len(returns))
    return float(total_growth ** (trading_days_per_year / n) - 1.0)


def _annualized_volatility(
    returns: pd.Series,
    trading_days_per_year: float,
) -> float:
    daily_vol = float(returns.std(ddof=1))
    return daily_vol * float(np.sqrt(trading_days_per_year))


def _sharpe_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float,
    trading_days_per_year: float,
) -> float:
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / trading_days_per_year
    excess = returns - daily_rf

    annual_excess = float(excess.mean()) * trading_days_per_year
    annual_vol = _annualized_volatility(returns, trading_days_per_year)

    return annual_excess / annual_vol if annual_vol > 1e-12 else 0.0


def _sortino_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float,
    trading_days_per_year: float,
) -> float:
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / trading_days_per_year
    excess = returns - daily_rf

    annual_excess = float(excess.mean()) * trading_days_per_year
    downside = excess[excess < 0.0]

    if len(downside) < 1:
        return 0.0 if annual_excess >= 0 else -float("inf")

    downside_vol = float(
        np.sqrt(np.mean(np.square(downside.values))) * np.sqrt(trading_days_per_year)
    )

    return annual_excess / downside_vol if downside_vol > 1e-12 else 0.0


def _calmar_ratio(
    annualized_return: float,
    max_drawdown: float,
) -> float:
    if max_drawdown >= 0:
        return float("inf") if annualized_return > 0 else 0.0

    return annualized_return / abs(max_drawdown)


def benchmark_statistics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> dict[str, float | int]:
    """strategyとbenchmarkの共通日についてalpha/beta等を計算する。

    alphaは単純OLSの日次interceptを252倍した近似値。
    統計的有意性を保証するものではない。
    """
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 3:
        return {
            "benchmark_observations": len(aligned),
            "benchmark_beta": 0.0,
            "benchmark_alpha_annualized": 0.0,
            "benchmark_correlation": 0.0,
            "active_return_annualized": 0.0,
            "tracking_error_annualized": 0.0,
            "information_ratio": 0.0,
        }

    strategy_arr = aligned["strategy"].to_numpy(dtype=float)
    benchmark_arr = aligned["benchmark"].to_numpy(dtype=float)

    beta = float(np.cov(strategy_arr, benchmark_arr)[0, 1] / np.var(benchmark_arr, ddof=1))
    daily_alpha = float(strategy_arr.mean() - beta * benchmark_arr.mean())

    active = strategy_arr - benchmark_arr
    active_annualized = float(active.mean()) * trading_days_per_year
    tracking_error = float(active.std(ddof=1)) * float(np.sqrt(trading_days_per_year))

    information_ratio = active_annualized / tracking_error if tracking_error > 0 else 0.0

    return {
        "benchmark_observations": len(aligned),
        "benchmark_beta": beta,
        "benchmark_alpha_annualized": (daily_alpha * trading_days_per_year),
        "benchmark_correlation": float(aligned["strategy"].corr(aligned["benchmark"])),
        "active_return_annualized": active_annualized,
        "tracking_error_annualized": tracking_error,
        "information_ratio": information_ratio,
    }


def summarize_research_performance(
    ledger: pd.DataFrame,
    *,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    benchmark_returns: pd.Series | None = None,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> dict:
    """研究用portfolio performance reportを返す。"""
    returns = ledger_returns(
        ledger,
        initial_capital=initial_capital,
    )

    if returns.empty:
        return {"error": "ledger is empty"}

    drawdown = drawdown_series_from_returns(returns)

    annual_return = _annualized_return(
        returns,
        trading_days_per_year,
    )
    annual_volatility = _annualized_volatility(
        returns,
        trading_days_per_year,
    )
    sharpe = _sharpe_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
    )
    sortino = _sortino_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
    )

    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    calmar = (
        annual_return / abs(max_drawdown)
        if max_drawdown < 0
        else float("inf")
        if annual_return > 0
        else 0.0
    )

    result: dict = {
        "observations": len(returns),
        "total_return": float((1.0 + returns).prod() - 1.0),
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "positive_day_ratio": float((returns > 0).mean()),
        "negative_day_ratio": float((returns < 0).mean()),
    }

    if "gross_exposure" in ledger.columns:
        result["average_gross_exposure"] = float(
            pd.to_numeric(
                ledger["gross_exposure"],
                errors="coerce",
            ).mean()
        )
        result["max_gross_exposure"] = float(
            pd.to_numeric(
                ledger["gross_exposure"],
                errors="coerce",
            ).max()
        )

    if "net_exposure" in ledger.columns:
        result["average_net_exposure"] = float(
            pd.to_numeric(
                ledger["net_exposure"],
                errors="coerce",
            ).mean()
        )

    if benchmark_returns is not None:
        stat = benchmark_statistics(
            returns,
            benchmark_returns,
            trading_days_per_year=trading_days_per_year,
        )
        result.update(stat)

    return result


def estimate_trade_turnover(
    trades: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    """日次売買代金とturnoverを推定する。

    turnover = 当日約定代金合計 / 前日NAV

    entryとexitを別々に計上するため、往復売買は両方含まれる。
    """
    columns = [
        "date",
        "traded_notional",
        "previous_nav",
        "turnover",
    ]

    if trades is None or trades.empty:
        return pd.DataFrame(columns=columns)

    required = {
        "entry_date",
        "exit_date",
        "entry",
        "exit",
        "qty",
    }
    missing = required - set(trades.columns)

    if missing:
        raise ValueError(f"trades missing columns: {sorted(missing)}")

    rows: list[dict] = []

    for _, trade in trades.iterrows():
        qty = int(trade["qty"])

        rows.append(
            {
                "date": pd.Timestamp(trade["entry_date"]),
                "notional": (float(trade["entry"]) * qty),
            }
        )
        rows.append(
            {
                "date": pd.Timestamp(trade["exit_date"]),
                "notional": (float(trade["exit"]) * qty),
            }
        )

    traded = pd.DataFrame(rows).groupby("date")["notional"].sum().rename("traded_notional")

    ledger_frame = ledger.copy()
    ledger_frame["date"] = pd.to_datetime(ledger_frame["date"])
    ledger_frame = (
        ledger_frame.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    )

    nav = pd.to_numeric(
        ledger_frame["nav"],
        errors="raise",
    )

    previous_nav = nav.shift(1)
    previous_nav.iloc[0] = float(initial_capital)

    result = pd.concat(
        [
            traded,
            previous_nav.rename("previous_nav"),
        ],
        axis=1,
        join="inner",
    )

    result["turnover"] = result["traded_notional"] / result["previous_nav"]

    return result.reset_index()[columns]


def summarize_capacity(
    trades: pd.DataFrame,
    *,
    participation_limits: Iterable[float] = (
        0.001,
        0.002,
        0.005,
        0.01,
    ),
) -> dict:
    """tradeのADV participationを集計する。"""
    if trades is None or trades.empty:
        return {"error": "trades is empty"}

    required = {
        "entry",
        "exit",
        "qty",
        "entry_adv",
        "exit_adv",
    }
    missing = required - set(trades.columns)

    if missing:
        raise ValueError(f"trades missing columns: {sorted(missing)}")

    frame = trades.copy()

    frame["entry_notional"] = pd.to_numeric(frame["entry"]) * pd.to_numeric(frame["qty"])
    frame["exit_notional"] = pd.to_numeric(frame["exit"]) * pd.to_numeric(frame["qty"])

    frame["entry_participation"] = frame["entry_notional"] / pd.to_numeric(frame["entry_adv"])
    frame["exit_participation"] = frame["exit_notional"] / pd.to_numeric(frame["exit_adv"])

    participation = pd.concat(
        [
            frame["entry_participation"],
            frame["exit_participation"],
        ],
        ignore_index=True,
    )

    participation = participation[np.isfinite(participation) & (participation >= 0)]

    if participation.empty:
        return {"error": "valid participation is empty"}

    result: dict[str, float | int] = {
        "execution_count": len(participation),
        "participation_mean": float(participation.mean()),
        "participation_median": float(participation.median()),
        "participation_p95": float(participation.quantile(0.95)),
        "participation_max": float(participation.max()),
    }

    for limit in participation_limits:
        key = f"fraction_above_{limit:.4f}".replace(".", "_")
        result[key] = float((participation > limit).mean())

    return result


def holm_adjust(
    p_values: Iterable[float],
) -> np.ndarray:
    """Holm法によるfamily-wise error rate補正。

    statsmodelsへの依存を増やさず実装する。
    """
    values = np.asarray(
        list(p_values),
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")

    if len(values) == 0:
        return values

    if (~np.isfinite(values)).any() or (values < 0).any() or (values > 1).any():
        raise ValueError("p_values must be finite and between 0 and 1")

    order = np.argsort(values)
    sorted_values = values[order]
    m = len(values)

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for i, p_value in enumerate(sorted_values):
        adjusted = (m - i) * p_value
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted
