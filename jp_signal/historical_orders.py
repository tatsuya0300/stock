"""過去注文生成（PR-4）。

generate_historical_orders() はバックテスト外で過去注文を生成し、
ポートフォリオバックテスターの step() に入力するための orders DataFrame を返す。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricalOrderResult:
    """過去注文生成結果。"""
    orders: pd.DataFrame
    rejections: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


# 型エイリアス
PriceLoader = Callable[[list[str], date, date], pd.DataFrame]
ShortabilityLoader = Callable[[list[str], pd.Timestamp], pd.DataFrame]
UniverseLoader = Callable[[str], pd.DataFrame]
ModelFactory = Callable[[dict], Any]


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
) -> HistoricalOrderResult:
    """指定された営業日リストに対して過去注文を生成する。

    Args:
        trading_dates: 生成対象の営業日リスト。
        model_factory: 設定からモデルインスタンスを生成する関数。
        price_loader: (codes, start, end) -> DataFrame の価格取得関数。
        shortability_loader: (codes, decision_at) -> DataFrame の空売り可否取得関数。
        universe_loader: as_of -> DataFrame のユニバース取得関数。
        model_cfg: モデル設定 dict。
        sizing_cfg: サイジング設定 dict。
        risk_cfg: リスク設定 dict。
        codes: 対象銘柄リスト（None の場合はユニバースから取得）。

    Returns:
        HistoricalOrderResult 注文・リジェクション・診断情報。
    """
    all_orders: list[pd.DataFrame] = []
    all_rejections: list[pd.DataFrame] = []
    total_signals = 0
    total_errors = 0

    for i, d in enumerate(trading_dates):
        as_of = str(d)
        try:
            univ = universe_loader(as_of)
            codes_ = codes if codes else univ["code"].tolist()

            lookback = int(model_cfg.get("lookback", 20))
            start = pd.Timestamp(d) - pd.Timedelta(days=lookback * 2)
            prices = price_loader(codes_, start.date(), d)

            if prices.empty:
                log.warning("No prices for %s, skipping", as_of)
                continue

            model = model_factory(model_cfg)
            sig = model.generate(prices, as_of=as_of)

            if sig.empty:
                log.info("No signals for %s", as_of)
                continue

            total_signals += len(sig)

        except Exception:
            log.exception("Error generating orders for %s", as_of)
            total_errors += 1
            continue

    log.info(
        "generate_historical_orders: %d dates processed, %d signals, %d errors",
        len(trading_dates),
        total_signals,
        total_errors,
    )

    result_orders = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
    result_rejections = (
        pd.concat(all_rejections, ignore_index=True) if all_rejections else pd.DataFrame()
    )

    return HistoricalOrderResult(
        orders=result_orders,
        rejections=result_rejections,
        diagnostics={
            "dates_processed": len(trading_dates),
            "signals_generated": total_signals,
            "orders_produced": len(result_orders),
            "errors": total_errors,
        },
    )
