"""シグナル → 発注への共通変換。

live pipeline と backtest が同一ロジックを通ることで
研究と本番の乖離（implementation shortfall）を減らす。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .adv import rolling_adv_before
from .calendar import previous_business_day
from .risk import (
    RiskConfig,
    RiskSelectionResult,
    apply_order_risk_limits,
    select_orders_with_reasons,
)
from .sizing import compute_size


def is_shortable_asof(
    shortability: pd.DataFrame | None,
    code: str,
    as_of: date | str,
) -> bool:
    """as_of 以前の直近スナップショットで売り可否を判定。

    欠損・未来参照は常に False（保守側）。
    """
    if shortability is None or shortability.empty:
        return False

    sub = shortability[shortability["code"].astype(str) == str(code)].copy()
    if sub.empty:
        return False

    asof_ts = pd.Timestamp(as_of)
    sub["date"] = pd.to_datetime(sub["date"])
    past = sub[sub["date"] <= asof_ts]
    if past.empty:
        return False

    latest = past.sort_values("date").iloc[-1]
    return int(latest["is_margin_lendable"]) == 1 and int(latest["short_restricted"]) == 0


def _ref_rows_before(prices: pd.DataFrame, as_of: date | str) -> pd.DataFrame:
    """寄前想定: as_of 当日終値は未確定のため前営業日以前の最終行を使う。"""
    as_of_d = pd.Timestamp(as_of).date()
    cutoff = previous_business_day(as_of_d).isoformat()
    prev = prices[prices["date"] <= cutoff].sort_values("date")
    if prev.empty:
        return pd.DataFrame()
    return prev.groupby("code").tail(1).set_index("code")


def signals_to_orders(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    as_of: date | str,
    sizing_cfg: dict,
    risk_cfg: RiskConfig,
    shortability: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    holding_days: int = 1,
    order_type: str = "MKT_OPEN",
    unit: int = 100,
    for_backtest: bool = False,
) -> pd.DataFrame:
    """シグナル DataFrame をリスク制限後の orders に変換する。

    Returns columns (最低限):
      code, side, qty, order_type, value_yen, score, shortable, ref_price
      (+ name, warn, date, limit_price, holding_days when for_backtest)
    """
    empty_cols = [
        "code",
        "name",
        "side",
        "order_type",
        "qty",
        "ref_price",
        "value_yen",
        "score",
        "warn",
        "shortable",
        "date",
        "limit_price",
        "holding_days",
    ]
    if signals is None or signals.empty or prices is None or prices.empty:
        return pd.DataFrame(columns=empty_cols)

    last_row = _ref_rows_before(prices, as_of)
    if last_row.empty:
        return pd.DataFrame(columns=empty_cols)

    adv_window = int(sizing_cfg.get("adv_window", 20))
    min_adv_periods = int(sizing_cfg.get("min_adv_periods", 1))
    require_full_adv_history = bool(sizing_cfg.get("require_full_adv_history", False))
    allow_single_day_turnover_fallback = bool(
        sizing_cfg.get("allow_single_day_turnover_fallback", True)
    )

    adv_series = rolling_adv_before(
        prices,
        as_of,
        window=adv_window,
        min_periods=min_adv_periods,
        strictly_before=True,
    )

    univ_idx = None
    if universe is not None and not universe.empty:
        univ_idx = universe.set_index("code")

    as_of_d = pd.Timestamp(as_of).date()
    rows: list[dict] = []

    for _, r in signals.iterrows():
        code = str(r["code"])
        if code not in last_row.index:
            continue

        ref = float(last_row.loc[code, "close"])
        adv = float(adv_series.get(code, float("nan")))

        if not np.isfinite(adv) or adv <= 0:
            if require_full_adv_history:
                continue
            if allow_single_day_turnover_fallback:
                adv = float(last_row.loc[code, "turnover"])
        qty, yen, warn = compute_size(
            adv,
            ref,
            float(sizing_cfg["adv_ratio"]),
            float(sizing_cfg["adv_ratio_cap"]),
            unit=unit,
            market_open_unit_cap=int(sizing_cfg.get("market_open_unit_cap", 50)),
            is_market_open_order=(order_type == "MKT_OPEN"),
        )
        if qty == 0:
            continue

        shortable = is_shortable_asof(shortability, code, as_of_d)

        name = ""
        if univ_idx is not None and code in univ_idx.index:
            name = str(univ_idx.loc[code, "name"])

        # name は live/BT 共通で常に付与（通知・DB用）
        row: dict = {
            "code": code,
            "name": name,
            "side": str(r["side"]).upper(),
            "order_type": order_type,
            "qty": qty,
            "ref_price": ref,
            "value_yen": yen,
            "score": float(r.get("score", 0)),
            "warn": warn,
            "shortable": shortable,
        }

        # BT専用列のみ for_backtest 時に付与
        if for_backtest:
            row["date"] = str(as_of_d)
            row["limit_price"] = r.get("limit_price", None)
            row["holding_days"] = holding_days

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    orders = pd.DataFrame(rows)

    # リスク制限
    orders = apply_order_risk_limits(orders, risk_cfg, score_col="score")

    return orders


def apply_order_risk_with_audit(
    orders: pd.DataFrame,
    risk_cfg: RiskConfig,
) -> RiskSelectionResult:
    """注文へリスク制限を適用し、採用・不採用を両方返す。"""
    return select_orders_with_reasons(
        orders,
        risk_cfg,
        score_col="score",
    )
