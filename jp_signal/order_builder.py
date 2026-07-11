"""シグナル → 発注への共通変換。

live pipeline と backtest が同一ロジックを通ることで
研究と本番の乖離（implementation shortfall）を減らす。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from . import normalize_code
from .adv import rolling_adv_before
from .calendar import previous_business_day
from .risk import (
    RiskConfig,
    RiskSelectionResult,
    select_orders_with_reasons,
)
from .shortability_pit import (
    decide_shortability,
)
from .sizing import compute_size


def is_shortable_asof(
    shortability: pd.DataFrame | None,
    code: str,
    as_of: date | str,
    max_age_calendar_days: int = 4,
) -> bool:
    """as_of以前の直近スナップショットで売り可否を判定する。

    条件:
      - 未来のスナップショットを参照しない
      - 最新スナップショットがmax_age_calendar_days以内
      - is_margin_lendable == 1
      - short_restricted == 0

    金曜日のスナップショットを月曜日に利用できるよう、
    デフォルトは4暦日とする。ただし本番では取得時刻を含む
    effective_at/fetched_at管理へ移行すること。
    """
    if max_age_calendar_days < 0:
        raise ValueError(f"max_age_calendar_days must be >= 0: {max_age_calendar_days}")

    if shortability is None or shortability.empty:
        return False

    # PITカラムが存在する場合、PIT実装を優先する
    pit_columns = {
        "effective_at",
        "fetched_at",
        "source",
        "short_type",
        "is_shortable",
        "short_restricted",
    }

    if pit_columns.issubset(shortability.columns):
        decision = decide_shortability(
            shortability,
            code=code,
            as_of=as_of,
            requested_short_type="system",
            max_age=pd.Timedelta(days=max_age_calendar_days),
        )
        return decision.is_shortable

    # 以下は既存legacy実装
    required = {
        "code",
        "date",
        "is_margin_lendable",
        "short_restricted",
    }
    missing = required - set(shortability.columns)

    if missing:
        return False

    sub = shortability[shortability["code"].astype(str) == str(code)].copy()

    if sub.empty:
        return False

    asof_ts = pd.Timestamp(as_of).normalize()
    sub["date"] = pd.to_datetime(
        sub["date"],
        errors="coerce",
    ).dt.normalize()
    sub = sub.dropna(subset=["date"])

    past = sub[sub["date"] <= asof_ts]
    if past.empty:
        return False

    latest = past.sort_values("date").iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    age_days = (asof_ts - latest_date).days

    if age_days > max_age_calendar_days:
        return False

    try:
        is_margin_lendable = int(latest["is_margin_lendable"])
        short_restricted = int(latest["short_restricted"])
    except (TypeError, ValueError):
        return False

    return is_margin_lendable == 1 and short_restricted == 0


def _ref_rows_before(prices: pd.DataFrame, as_of: date | str) -> pd.DataFrame:
    """寄前時点で利用可能な直近価格を銘柄別に返す。

    as_of当日の価格は使用せず、前営業日以前のデータだけを利用する。
    date列は文字列・Timestampのどちらも受け付ける。
    """
    required = {"code", "date"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {sorted(missing)}")

    as_of_d = pd.Timestamp(as_of).date()
    cutoff = pd.Timestamp(previous_business_day(as_of_d))

    previous = prices.copy()
    previous["code"] = (
        previous["code"]
        .astype(str)
        .str.strip()
    )
    previous["date"] = pd.to_datetime(
        previous["date"],
        errors="coerce",
    ).dt.normalize()

    previous = previous.dropna(subset=["date"])
    previous = previous[
        previous["date"] <= cutoff
    ].sort_values("date")

    if previous.empty:
        return pd.DataFrame()

    return previous.groupby("code").tail(1).set_index("code")


_ORDER_COLUMNS = [
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

_LIVE_ORDER_COLUMNS = [
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
]


def _order_columns(*, for_backtest: bool) -> list[str]:
    """live/BTの注文出力スキーマを返す。

    live注文にはバックテスト専用列を含めない。
    これにより、for_backtest=Falseで存在しない列を選択して
    KeyErrorになることを防ぐ。
    """
    if for_backtest:
        return list(_ORDER_COLUMNS)
    return list(_LIVE_ORDER_COLUMNS)


@dataclass(frozen=True)
class OrderBuildResult:
    selected: pd.DataFrame
    rejected: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RejectionRecord:
    stage: str
    reason: str
    code: str
    side: str
    score: float | None = None
    qty: int | None = None
    ref_price: float | None = None
    value_yen: float | None = None
    name: str = ""
    warn: str = ""
    shortable: bool | None = None


_REJECTION_COLUMNS = [
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
    "stage",
    "reason",
]


def _empty_orders(*, for_backtest: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        columns=_order_columns(
            for_backtest=for_backtest,
        )
    )


def _empty_rejections() -> pd.DataFrame:
    return pd.DataFrame(columns=_REJECTION_COLUMNS)


def build_orders_with_audit(
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
    shortability_max_age_days: int = 4,
    decision_at: str | pd.Timestamp | None = None,
) -> OrderBuildResult:
    """シグナルを注文へ変換し、採用・不採用を全て返す。"""
    if signals is None or signals.empty:
        return OrderBuildResult(
            selected=_empty_orders(
                for_backtest=for_backtest,
            ),
            rejected=_empty_rejections(),
            diagnostics={
                "signal_count": 0,
                "candidate_count": 0,
                "selected_count": 0,
                "rejected_count": 0,
            },
        )

    if prices is None or prices.empty:
        rejected = signals.copy()
        rejected["stage"] = "REFERENCE_DATA"
        rejected["reason"] = "EMPTY_PRICES"
        for col in _REJECTION_COLUMNS:
            if col not in rejected.columns:
                rejected[col] = None
        return OrderBuildResult(
            selected=_empty_orders(
                for_backtest=for_backtest,
            ),
            rejected=rejected[_REJECTION_COLUMNS],
            diagnostics={
                "signal_count": len(signals),
                "candidate_count": 0,
                "selected_count": 0,
                "rejected_count": len(rejected),
            },
        )

    last_row = _ref_rows_before(prices, as_of)

    if last_row.empty:
        rejected = signals.copy()
        rejected["stage"] = "REFERENCE_DATA"
        rejected["reason"] = "NO_REFERENCE_DATE"
        for col in _REJECTION_COLUMNS:
            if col not in rejected.columns:
                rejected[col] = None
        return OrderBuildResult(
            selected=_empty_orders(
                for_backtest=for_backtest,
            ),
            rejected=rejected[_REJECTION_COLUMNS],
            diagnostics={
                "signal_count": len(signals),
                "candidate_count": 0,
                "selected_count": 0,
                "rejected_count": len(rejected),
            },
        )

    adv_window = int(sizing_cfg.get("adv_window", 20))
    min_adv_periods = int(sizing_cfg.get("min_adv_periods", 1))
    require_full_adv_history = bool(sizing_cfg.get("require_full_adv_history", True))
    allow_single_day_turnover_fallback = bool(
        sizing_cfg.get("allow_single_day_turnover_fallback", False)
    )

    adv_series = rolling_adv_before(
        prices,
        as_of,
        window=adv_window,
        min_periods=min_adv_periods,
        strictly_before=True,
    )

    univ_idx: pd.DataFrame | None = None
    if universe is not None and not universe.empty:
        univ_idx = universe.set_index("code")

    as_of_d = pd.Timestamp(as_of).date()

    candidate_rows: list[dict] = []
    rejected_rows: list[dict] = []

    def _reject(
        signal: pd.Series,
        *,
        stage: str,
        reason: str,
    ) -> None:
        row = signal.to_dict()
        # 欠損列を埋める
        for col in _REJECTION_COLUMNS:
            if col not in row:
                row[col] = None
        row["stage"] = stage
        row["reason"] = reason
        rejected_rows.append(row)

    for _, signal in signals.iterrows():
        code = normalize_code(signal.get("code", ""))
        side = str(signal.get("side", "")).upper()

        if not code:
            _reject(signal, stage="VALIDATION", reason="EMPTY_CODE")
            continue

        if side not in {"BUY", "SELL"}:
            _reject(signal, stage="VALIDATION", reason="INVALID_SIDE")
            continue

        if code not in last_row.index:
            _reject(signal, stage="REFERENCE_DATA", reason="NO_REFERENCE_PRICE")
            continue

        ref = float(pd.to_numeric(last_row.loc[code, "close"], errors="coerce"))

        if not np.isfinite(ref) or ref <= 0:
            _reject(signal, stage="REFERENCE_DATA", reason="INVALID_REFERENCE_PRICE")
            continue

        adv = float(adv_series.get(code, float("nan")))
        adv_stage = "ADV"

        if not np.isfinite(adv) or adv <= 0:
            if require_full_adv_history:
                _reject(signal, stage=adv_stage, reason="ADV_UNAVAILABLE")
                continue
            if allow_single_day_turnover_fallback:
                adv = float(
                    pd.to_numeric(
                        last_row.loc[code, "turnover"],
                        errors="coerce",
                    )
                )
                if not np.isfinite(adv) or adv <= 0:
                    _reject(signal, stage=adv_stage, reason="ADV_FALLBACK_ZERO")
                    continue

        enforce_cap = bool(sizing_cfg.get("enforce_market_open_unit_cap", False))
        qty, yen, warn = compute_size(
            adv,
            ref,
            float(sizing_cfg["adv_ratio"]),
            float(sizing_cfg["adv_ratio_cap"]),
            unit=unit,
            market_open_unit_cap=int(sizing_cfg.get("market_open_unit_cap", 50)),
            is_market_open_order=(order_type == "MKT_OPEN"),
            enforce_market_open_unit_cap=enforce_cap,
        )

        if qty == 0:
            _reject(signal, stage="SIZING", reason="QTY_ZERO")
            continue

        shortability_as_of = decision_at if decision_at is not None else as_of_d

        shortable = is_shortable_asof(
            shortability,
            code,
            shortability_as_of,
            max_age_calendar_days=shortability_max_age_days,
        )

        name = ""
        if univ_idx is not None and code in univ_idx.index:
            name = str(univ_idx.loc[code, "name"])

        candidate_row: dict = {
            "code": code,
            "name": name,
            "side": side,
            "order_type": order_type,
            "qty": qty,
            "ref_price": ref,
            "value_yen": yen,
            "risk_value_yen": yen
            * (1.0 + float(sizing_cfg.get("reference_price_buffer_ratio", 0.0))),
            "score": float(signal.get("score", 0)),
            "warn": warn,
            "shortable": shortable,
        }

        if for_backtest:
            candidate_row["date"] = str(as_of_d)
            candidate_row["limit_price"] = signal.get("limit_price", None)
            candidate_row["holding_days"] = holding_days

        candidate_rows.append(candidate_row)

    if not candidate_rows:
        return OrderBuildResult(
            selected=_empty_orders(
                for_backtest=for_backtest,
            ),
            rejected=pd.DataFrame(rejected_rows) if rejected_rows else _empty_rejections(),
            diagnostics={
                "signal_count": len(signals),
                "candidate_count": 0,
                "selected_count": 0,
                "rejected_count": len(rejected_rows),
            },
        )

    orders_df = pd.DataFrame(candidate_rows)

    # リスク制限
    risk_result = select_orders_with_reasons(
        orders_df,
        risk_cfg,
        score_col="score",
    )

    selected = risk_result.selected
    for _, rejected_candidate in risk_result.rejected.iterrows():
        rec = rejected_candidate.to_dict()
        for col in _REJECTION_COLUMNS:
            if col not in rec:
                rec[col] = None
        rec["stage"] = "RISK_LIMIT"
        rec["reason"] = str(rejected_candidate.get("reason", "RISK_REJECTED"))
        rejected_rows.append(rec)

    # live/BTごとの出力スキーマで返す。
    output_columns = _order_columns(
        for_backtest=for_backtest,
    )

    if selected.empty:
        selected_out = _empty_orders(
            for_backtest=for_backtest,
        )
    else:
        missing_output_columns = set(output_columns) - set(selected.columns)

        if missing_output_columns:
            raise RuntimeError(
                f"selected orders missing output columns: {sorted(missing_output_columns)}"
            )

        selected_out = selected[output_columns].copy()

    rejected_out = (
        pd.DataFrame(rejected_rows)[_REJECTION_COLUMNS] if rejected_rows else _empty_rejections()
    )

    return OrderBuildResult(
        selected=selected_out,
        rejected=rejected_out,
        diagnostics={
            "signal_count": len(signals),
            "candidate_count": len(candidate_rows),
            "selected_count": len(selected),
            "rejected_count": len(rejected_rows),
        },
    )


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
    shortability_max_age_days: int = 4,
    decision_at: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """後方互換API。監査情報が必要ならbuild_orders_with_auditを使用する。"""
    result = build_orders_with_audit(
        signals,
        prices,
        as_of=as_of,
        sizing_cfg=sizing_cfg,
        risk_cfg=risk_cfg,
        shortability=shortability,
        universe=universe,
        holding_days=holding_days,
        order_type=order_type,
        unit=unit,
        for_backtest=for_backtest,
        shortability_max_age_days=shortability_max_age_days,
        decision_at=decision_at,
    )
    return result.selected


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
