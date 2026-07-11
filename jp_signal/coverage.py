"""データcoverage検証。

クロスセクショナル戦略では、部分的なデータ欠損がランキング母集団を歪める。
そのため注文生成前に、価格・lookback・turnoverのcoverageを検証する。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .calendar import previous_business_day


@dataclass(frozen=True)
class CoverageThresholds:
    price_coverage_min: float = 0.99
    lookback_coverage_min: float = 0.95
    turnover_coverage_min: float = 0.99
    hard_fail: bool = True

    @classmethod
    def from_config(cls, cfg: dict | None) -> CoverageThresholds:
        d = cfg or {}
        return cls(
            price_coverage_min=float(d.get("price_coverage_min", 0.99)),
            lookback_coverage_min=float(d.get("lookback_coverage_min", 0.95)),
            turnover_coverage_min=float(d.get("turnover_coverage_min", 0.99)),
            hard_fail=bool(d.get("hard_fail", True)),
        )


@dataclass(frozen=True)
class CoverageReport:
    as_of: str
    target_date: str
    expected_universe_count: int
    price_available_count: int
    lookback_available_count: int
    turnover_available_count: int
    price_coverage: float
    lookback_coverage: float
    turnover_coverage: float
    ok: bool
    failed_reasons: tuple[str, ...]

    def to_message(self) -> str:
        reasons = ", ".join(self.failed_reasons) if self.failed_reasons else "none"
        return (
            f"coverage as_of={self.as_of} target_date={self.target_date} "
            f"expected={self.expected_universe_count} "
            f"price={self.price_available_count}({self.price_coverage:.3f}) "
            f"lookback={self.lookback_available_count}({self.lookback_coverage:.3f}) "
            f"turnover={self.turnover_available_count}({self.turnover_coverage:.3f}) "
            f"ok={self.ok} reasons={reasons}"
        )


def validate_daily_coverage(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    as_of: date | str | pd.Timestamp,
    lookback: int,
    adv_window: int,
    min_adv_periods: int,
    thresholds: CoverageThresholds,
) -> CoverageReport:
    """注文生成前のcoverageを検証する。

    寄前戦略を想定し、as_of当日ではなく前営業日をtarget_dateにする。
    """
    if universe is None or universe.empty:
        return CoverageReport(
            as_of=str(pd.Timestamp(as_of).date()),
            target_date="",
            expected_universe_count=0,
            price_available_count=0,
            lookback_available_count=0,
            turnover_available_count=0,
            price_coverage=0.0,
            lookback_coverage=0.0,
            turnover_coverage=0.0,
            ok=False,
            failed_reasons=("EMPTY_UNIVERSE",),
        )

    if prices is None or prices.empty:
        asof_date = pd.Timestamp(as_of).date()
        target_date = previous_business_day(asof_date)
        n = len(universe)
        return CoverageReport(
            as_of=asof_date.isoformat(),
            target_date=target_date.isoformat(),
            expected_universe_count=n,
            price_available_count=0,
            lookback_available_count=0,
            turnover_available_count=0,
            price_coverage=0.0,
            lookback_coverage=0.0,
            turnover_coverage=0.0,
            ok=False,
            failed_reasons=("EMPTY_PRICES",),
        )

    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if adv_window < 1:
        raise ValueError("adv_window must be >= 1")
    if min_adv_periods < 1:
        raise ValueError("min_adv_periods must be >= 1")
    if min_adv_periods > adv_window:
        raise ValueError("min_adv_periods must be <= adv_window")

    asof_date = pd.Timestamp(as_of).date()
    target_date = previous_business_day(asof_date)
    target_s = str(target_date)

    expected_n = len(universe)

    x = prices.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values(["code", "date"])

    # target_date の行情
    target_rows = x[x["date"] == pd.Timestamp(target_date)]

    if target_rows.empty:
        return CoverageReport(
            as_of=str(asof_date),
            target_date=target_s,
            expected_universe_count=expected_n,
            price_available_count=0,
            lookback_available_count=0,
            turnover_available_count=0,
            price_coverage=0.0,
            lookback_coverage=0.0,
            turnover_coverage=0.0,
            ok=False,
            failed_reasons=("NO_TARGET_DATE_DATA",),
        )

    # OHLC が正の銘柄
    price_ok = target_rows[
        (target_rows["open"] > 0)
        & (target_rows["high"] > 0)
        & (target_rows["low"] > 0)
        & (target_rows["close"] > 0)
    ]
    price_codes = set(price_ok["code"])

    turnover_ok = target_rows[np.isfinite(target_rows["turnover"]) & (target_rows["turnover"] > 0)]
    turnover_codes = set(turnover_ok["code"])

    # target_date 以前に lookback 営業日以上があるか
    history = x[x["date"] < pd.Timestamp(target_date)]
    required_bars = max(lookback, min_adv_periods)
    lookback_codes = set(
        history.groupby("code")
        .filter(lambda g: len(g) >= required_bars)["code"]
        .unique()
        .astype(str)
    )

    price_count = len(price_codes)
    turnover_count = len(turnover_codes)
    lookback_count = len(lookback_codes)

    denom = max(expected_n, 1)
    price_coverage = price_count / denom
    turnover_coverage = turnover_count / denom
    lookback_coverage = lookback_count / denom

    failed: list[str] = []
    if price_coverage < thresholds.price_coverage_min:
        failed.append("PRICE_COVERAGE_LOW")
    if turnover_coverage < thresholds.turnover_coverage_min:
        failed.append("TURNOVER_COVERAGE_LOW")
    if lookback_coverage < thresholds.lookback_coverage_min:
        failed.append("LOOKBACK_COVERAGE_LOW")

    ok = len(failed) == 0

    return CoverageReport(
        as_of=str(asof_date),
        target_date=target_s,
        expected_universe_count=expected_n,
        price_available_count=price_count,
        lookback_available_count=lookback_count,
        turnover_available_count=turnover_count,
        price_coverage=price_coverage,
        lookback_coverage=lookback_coverage,
        turnover_coverage=turnover_coverage,
        ok=ok,
        failed_reasons=tuple(failed),
    )
