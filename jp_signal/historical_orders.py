"""過去注文生成。

各営業日の判断時点で利用可能な価格、ユニバース、売建可否を使い、
live pipelineと同じorder_builderを通してバックテスト用注文を生成する。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from .order_builder import build_orders_with_audit
from .risk import risk_config_from_dict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricalOrderResult:
    """過去注文生成結果。"""

    orders: pd.DataFrame
    rejections: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


PriceLoader = Callable[[list[str], date, date], pd.DataFrame]
ShortabilityLoader = Callable[[list[str], pd.Timestamp], pd.DataFrame]
UniverseLoader = Callable[[str], pd.DataFrame]
ModelFactory = Callable[[dict], Any]


def _decision_at(
    trading_date: date,
    decision_time: str,
) -> pd.Timestamp:
    """取引日の判断時刻をJST timezone-aware Timestampで返す。"""
    timestamp = pd.Timestamp(
        f"{trading_date.isoformat()} {decision_time}"
    )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Tokyo")

    return timestamp


def generate_historical_orders(
    *,
    trading_dates: list[date],
    model_factory: ModelFactory,
    price_loader: PriceLoader,
    shortability_loader: ShortabilityLoader | None = None,
    universe_loader: UniverseLoader,
    model_cfg: dict,
    sizing_cfg: dict,
    risk_cfg: dict,
    codes: list[str] | None = None,
    holding_days: int = 1,
    unit: int = 100,
    decision_time: str = "08:15",
    order_type: str = "MKT_OPEN",
    shortability_max_age_days: int = 4,
    fail_fast: bool = True,
) -> HistoricalOrderResult:
    """指定された営業日に対して過去注文を生成する。

    重要:
      price_loaderは、呼び出された日付時点で利用可能だった価格だけを
      返す責任を持つ。point-in-timeデータを使う場合は、呼び出し側で
      available_at制約を適用すること。

    Args:
        trading_dates:
            注文生成対象の営業日。
        model_factory:
            model_cfgからSignalModelを生成する関数。
        price_loader:
            (codes, start, end) -> prices DataFrame。
        shortability_loader:
            (codes, decision_at) -> shortability DataFrame。
        universe_loader:
            as_of文字列から当時のユニバースを返す関数。
        fail_fast:
            Trueなら1日でも処理失敗した時点で例外を送出する。
            部分的に欠損したBTを誤って採用しないため、デフォルトTrue。

    Returns:
        HistoricalOrderResult。
    """
    if holding_days < 1:
        raise ValueError(
            f"holding_days must be >= 1: {holding_days}"
        )

    if unit < 1:
        raise ValueError(
            f"unit must be >= 1: {unit}"
        )

    all_orders: list[pd.DataFrame] = []
    all_rejections: list[pd.DataFrame] = []

    total_signals = 0
    total_errors = 0
    dates_with_prices = 0
    dates_with_signals = 0
    dates_with_orders = 0

    risk = risk_config_from_dict(risk_cfg)

    model_lookback = int(
        model_cfg.get("lookback", 20)
    )
    adv_window = int(
        sizing_cfg.get("adv_window", 20)
    )
    min_adv_periods = int(
        sizing_cfg.get("min_adv_periods", 1)
    )

    warmup_bars = max(
        model_lookback + 1,
        adv_window,
        min_adv_periods,
    )

    # 営業日数を十分に包含する暦日バッファ。
    warmup_calendar_days = max(
        warmup_bars * 3,
        30,
    )

    for trading_date in sorted(set(trading_dates)):
        as_of = trading_date.isoformat()

        try:
            universe = universe_loader(as_of)

            if universe is None or universe.empty:
                log.warning(
                    "Empty universe for %s",
                    as_of,
                )
                continue

            if "code" not in universe.columns:
                raise ValueError(
                    "universe missing column: code"
                )

            if codes is None:
                active_codes = (
                    universe["code"]
                    .astype(str)
                    .str.strip()
                    .drop_duplicates()
                    .tolist()
                )
            else:
                allowed = set(
                    universe["code"]
                    .astype(str)
                    .str.strip()
                )
                active_codes = [
                    str(code).strip()
                    for code in codes
                    if str(code).strip() in allowed
                ]

            if not active_codes:
                log.warning(
                    "No active codes for %s",
                    as_of,
                )
                continue

            start_date = (
                pd.Timestamp(trading_date)
                - pd.Timedelta(
                    days=warmup_calendar_days
                )
            ).date()

            prices = price_loader(
                active_codes,
                start_date,
                trading_date,
            )

            if prices is None or prices.empty:
                log.warning(
                    "No prices for %s",
                    as_of,
                )
                continue

            dates_with_prices += 1

            decision_at = _decision_at(
                trading_date,
                decision_time,
            )

            shortability = None

            if shortability_loader is not None:
                loaded_shortability = shortability_loader(
                    active_codes,
                    decision_at,
                )

                if (
                    loaded_shortability is not None
                    and not loaded_shortability.empty
                ):
                    shortability = loaded_shortability

            model = model_factory(model_cfg)

            signals = model.generate(
                prices,
                as_of=as_of,
            )

            if signals is None or signals.empty:
                log.info(
                    "No signals for %s",
                    as_of,
                )
                continue

            dates_with_signals += 1
            total_signals += len(signals)

            build_result = build_orders_with_audit(
                signals,
                prices,
                as_of=trading_date,
                decision_at=decision_at,
                sizing_cfg=sizing_cfg,
                risk_cfg=risk,
                shortability=shortability,
                universe=universe,
                holding_days=holding_days,
                order_type=order_type,
                unit=unit,
                for_backtest=True,
                shortability_max_age_days=(
                    shortability_max_age_days
                ),
            )

            if not build_result.selected.empty:
                all_orders.append(
                    build_result.selected
                )
                dates_with_orders += 1

            if not build_result.rejected.empty:
                rejected = (
                    build_result.rejected.copy()
                )
                rejected["rejection_date"] = as_of
                all_rejections.append(rejected)

        except Exception:
            total_errors += 1

            log.exception(
                "Historical order generation failed: %s",
                as_of,
            )

            if fail_fast:
                raise

    orders = (
        pd.concat(
            all_orders,
            ignore_index=True,
        )
        if all_orders
        else pd.DataFrame()
    )

    rejections = (
        pd.concat(
            all_rejections,
            ignore_index=True,
        )
        if all_rejections
        else pd.DataFrame()
    )

    if not orders.empty:
        orders = (
            orders
            .sort_values(
                ["date", "code", "side"],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    log.info(
        "generate_historical_orders: "
        "dates=%d prices=%d signals_dates=%d "
        "order_dates=%d signals=%d orders=%d "
        "rejections=%d errors=%d",
        len(trading_dates),
        dates_with_prices,
        dates_with_signals,
        dates_with_orders,
        total_signals,
        len(orders),
        len(rejections),
        total_errors,
    )

    return HistoricalOrderResult(
        orders=orders,
        rejections=rejections,
        diagnostics={
            "dates_processed": len(trading_dates),
            "dates_with_prices": dates_with_prices,
            "dates_with_signals": dates_with_signals,
            "dates_with_orders": dates_with_orders,
            "signals_generated": total_signals,
            "orders_produced": len(orders),
            "rejections_produced": len(rejections),
            "errors": total_errors,
        },
    )
