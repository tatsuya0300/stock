"""MILP ベースリスク最適化（PR-5）。

scipy.optimize.milp を使ってポートフォリオ制約のもとで
注文選択を最適化する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .risk import RiskConfig

log = logging.getLogger(__name__)


@dataclass
class OptimizationDiagnostics:
    """最適化の診断情報。"""
    status: int
    fun: float
    message: str
    n_candidates: int = 0
    n_selected: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def optimize_orders(
    candidates: list[dict],
    risk: RiskConfig,
    *,
    existing_long: float = 0.0,
    existing_short: float = 0.0,
) -> tuple[list[dict], OptimizationDiagnostics]:
    """scipy.optimize.milp を使って注文選択を最適化する。

    Args:
        candidates: 注文候補のリスト。各 dict は code, side, score, entry_value を持つ。
        risk: リスク設定。
        existing_long: 既存ロングエクスポージャー。
        existing_short: 既存ショートエクスポージャー。

    Returns:
        (selected_orders, diagnostics) のタプル。
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        log.warning("scipy.optimize.milp not available, falling back to greedy")
        return _greedy_fallback(candidates, risk, existing_long, existing_short)

    n = len(candidates)
    if n == 0:
        return [], OptimizationDiagnostics(status=0, fun=0.0, message="no candidates")

    entry_values = np.array([float(c.get("entry_value", 0)) for c in candidates], dtype=np.float64)
    scores = np.array([float(c.get("score", 0)) for c in candidates], dtype=np.float64)
    sides = np.array([1 if c["side"] == "BUY" else -1 for c in candidates], dtype=np.float64)

    # 目的関数: 最大化 score (milp は最小化なので -score)
    c_obj = -scores

    # 線形制約
    constraints = []

    # max_orders_per_day
    constraints.append(
        LinearConstraint(
            np.ones((1, n)),
            lb=0,
            ub=risk.max_orders_per_day,
        )
    )

    # max_gross_exposure
    gross_coeff = entry_values.reshape(1, -1)
    constraints.append(
        LinearConstraint(
            gross_coeff,
            lb=-np.inf,
            ub=risk.max_gross_exposure_yen - existing_long - existing_short,
        )
    )

    # max_long_exposure
    long_coeff = np.maximum(entry_values * (sides > 0), 0).reshape(1, -1)
    constraints.append(
        LinearConstraint(
            long_coeff,
            lb=-np.inf,
            ub=risk.max_long_exposure_yen - existing_long,
        )
    )

    # max_short_exposure
    short_coeff = np.maximum(entry_values * (sides < 0), 0).reshape(1, -1)
    constraints.append(
        LinearConstraint(
            short_coeff,
            lb=-np.inf,
            ub=risk.max_short_exposure_yen - existing_short,
        )
    )

    # バイナリ変数（選択/非選択）
    bounds = Bounds(lb=0, ub=1)

    integrality = np.ones(n, dtype=int)  # 全て整数（バイナリ）

    try:
        result = milp(
            c=c_obj,
            constraints=constraints,
            bounds=bounds,
            integrality=integrality,
        )
    except Exception as e:
        log.warning("MILP optimization failed: %s, falling back to greedy", e)
        return _greedy_fallback(candidates, risk, existing_long, existing_short)

    if not result.success:
        log.warning("MILP optimization status=%d: %s", result.status, result.message)
        return _greedy_fallback(candidates, risk, existing_long, existing_short)

    selected_indices = np.where(result.x > 0.5)[0]
    selected = [candidates[i] for i in selected_indices]

    diagnostics = OptimizationDiagnostics(
        status=result.status,
        fun=float(result.fun),
        message=str(result.message),
        n_candidates=n,
        n_selected=len(selected),
        details={
            "solver": "scipy.milp",
            "n_variables": n,
            "n_constraints": len(constraints),
        },
    )

    log.info("optimize_orders: selected %d/%d candidates (status=%d)", len(selected), n, result.status)
    return selected, diagnostics


def _greedy_fallback(
    candidates: list[dict],
    risk: RiskConfig,
    existing_long: float,
    existing_short: float,
) -> tuple[list[dict], OptimizationDiagnostics]:
    """scipy が利用不可の場合のグリーディフォールバック。"""
    selected: list[dict] = []
    long_value = existing_long
    short_value = existing_short

    sorted_candidates = sorted(
        candidates,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )

    for c in sorted_candidates:
        if len(selected) >= risk.max_orders_per_day:
            break
        value = float(c.get("entry_value", 0))
        if c["side"] == "BUY":
            if long_value + value > risk.max_long_exposure_yen:
                continue
            long_value += value
        else:
            if short_value + value > risk.max_short_exposure_yen:
                continue
            short_value += value
        if long_value + short_value > risk.max_gross_exposure_yen:
            continue
        selected.append(c)

    return selected, OptimizationDiagnostics(
        status=-1,
        fun=0.0,
        message="greedy fallback (scipy milp unavailable)",
        n_candidates=len(candidates),
        n_selected=len(selected),
    )
