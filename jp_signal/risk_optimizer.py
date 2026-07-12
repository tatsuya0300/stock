"""MILPベース注文選択。

既存エクスポージャーを含めて以下を制約する。

- order count
- single-name exposure
- long exposure
- short exposure
- gross exposure
- net exposure
- require both sides
- confirmed shortability
- duplicate code
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .risk import RiskConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationDiagnostics:
    status: int
    fun: float
    message: str
    n_candidates: int = 0
    n_selected: int = 0
    details: dict[str, Any] = field(
        default_factory=dict
    )


def _normalize_candidates(
    candidates: list[dict],
    risk: RiskConfig,
) -> tuple[list[dict], list[dict]]:
    valid: list[dict] = []
    rejected: list[dict] = []
    seen_codes: set[str] = set()

    ordered = sorted(
        candidates,
        key=lambda row: float(
            row.get("score", 0.0)
        ),
        reverse=True,
    )

    for candidate in ordered:
        row = dict(candidate)
        code = str(
            row.get("code", "")
        ).strip()
        side = str(
            row.get("side", "")
        ).upper()

        try:
            raw_value: Any = row.get(
                "entry_value",
                row.get(
                    "risk_value_yen",
                    row.get(
                        "value_yen",
                        0.0,
                    ),
                ),
            )
            value = float(raw_value)
        except (TypeError, ValueError):
            value = float("nan")

        reason = None

        if not code:
            reason = "EMPTY_CODE"
        elif code in seen_codes:
            reason = "DUPLICATE_CODE"
        elif side not in {"BUY", "SELL"}:
            reason = "INVALID_SIDE"
        elif not np.isfinite(value) or value <= 0:
            reason = "INVALID_VALUE"
        elif (
            value
            > risk.max_single_name_exposure_yen
        ):
            reason = "SINGLE_NAME_LIMIT"
        elif (
            side == "SELL"
            and not risk.allow_short_without_confirmed_shortability
            and not bool(
                row.get("shortable", False)
            )
        ):
            reason = "NOT_SHORTABLE"

        if reason is not None:
            row["reason"] = reason
            rejected.append(row)
            continue

        seen_codes.add(code)

        row["code"] = code
        row["side"] = side
        row["_risk_value"] = value
        valid.append(row)

    return valid, rejected


def optimize_orders(
    candidates: list[dict],
    risk: RiskConfig,
    *,
    existing_long: float = 0.0,
    existing_short: float = 0.0,
    gross_limit_override: float | None = None,
) -> tuple[
    list[dict],
    list[dict],
    OptimizationDiagnostics,
]:
    valid, rejected = _normalize_candidates(
        candidates,
        risk,
    )

    if not valid:
        return (
            [],
            rejected,
            OptimizationDiagnostics(
                status=0,
                fun=0.0,
                message="no valid candidates",
                n_candidates=len(candidates),
                n_selected=0,
            ),
        )

    if risk.selection_method == "greedy":
        selected, additional_rejected = (
            _greedy_select(
                valid,
                risk,
                existing_long=existing_long,
                existing_short=existing_short,
                gross_limit_override=(
                    gross_limit_override
                ),
            )
        )

        return (
            selected,
            rejected + additional_rejected,
            OptimizationDiagnostics(
                status=0,
                fun=0.0,
                message="greedy",
                n_candidates=len(candidates),
                n_selected=len(selected),
                details={
                    "selector": "greedy",
                },
            ),
        )

    try:
        from scipy.optimize import (
            Bounds,
            LinearConstraint,
            milp,
        )
    except ImportError as exc:
        if not risk.allow_optimizer_fallback:
            raise RuntimeError(
                "selection_method=milpですが "
                "scipy.optimize.milpを利用できません"
            ) from exc

        selected, additional_rejected = (
            _greedy_select(
                valid,
                risk,
                existing_long=existing_long,
                existing_short=existing_short,
                gross_limit_override=(
                    gross_limit_override
                ),
            )
        )

        return (
            selected,
            rejected + additional_rejected,
            OptimizationDiagnostics(
                status=-1,
                fun=0.0,
                message="MILP unavailable; greedy fallback",
                n_candidates=len(candidates),
                n_selected=len(selected),
            ),
        )

    n = len(valid)

    values = np.asarray(
        [
            float(row["_risk_value"])
            for row in valid
        ],
        dtype=float,
    )
    scores = np.asarray(
        [
            float(row.get("score", 0.0))
            for row in valid
        ],
        dtype=float,
    )
    buy_mask = np.asarray(
        [
            row["side"] == "BUY"
            for row in valid
        ],
        dtype=float,
    )
    sell_mask = np.asarray(
        [
            row["side"] == "SELL"
            for row in valid
        ],
        dtype=float,
    )

    gross_limit = (
        risk.max_gross_exposure_yen
        if gross_limit_override is None
        else min(
            risk.max_gross_exposure_yen,
            gross_limit_override,
        )
    )

    remaining_gross = (
        gross_limit
        - existing_long
        - existing_short
    )
    remaining_long = (
        risk.max_long_exposure_yen
        - existing_long
    )
    remaining_short = (
        risk.max_short_exposure_yen
        - existing_short
    )

    if min(
        remaining_gross,
        remaining_long,
        remaining_short,
    ) < 0:
        return (
            [],
            rejected,
            OptimizationDiagnostics(
                status=2,
                fun=0.0,
                message=(
                    "existing exposure already "
                    "violates limits"
                ),
                n_candidates=len(candidates),
                n_selected=0,
            ),
        )

    constraints = [
        LinearConstraint(
            np.ones((1, n)),
            lb=0,
            ub=risk.max_orders_per_day,
        ),
        LinearConstraint(
            values.reshape(1, -1),
            lb=0,
            ub=remaining_gross,
        ),
        LinearConstraint(
            (values * buy_mask).reshape(
                1,
                -1,
            ),
            lb=0,
            ub=remaining_long,
        ),
        LinearConstraint(
            (values * sell_mask).reshape(
                1,
                -1,
            ),
            lb=0,
            ub=remaining_short,
        ),
    ]

    signed_values = (
        values * (buy_mask - sell_mask)
    )
    existing_net = (
        existing_long - existing_short
    )

    constraints.append(
        LinearConstraint(
            signed_values.reshape(1, -1),
            lb=(
                -risk.max_net_exposure_yen
                - existing_net
            ),
            ub=(
                risk.max_net_exposure_yen
                - existing_net
            ),
        )
    )

    if risk.require_both_sides:
        if not buy_mask.any() or not sell_mask.any():
            for row in valid:
                rejected_row = dict(row)
                rejected_row["reason"] = (
                    "REQUIRE_BOTH_SIDES"
                )
                rejected.append(
                    rejected_row
                )

            return (
                [],
                rejected,
                OptimizationDiagnostics(
                    status=2,
                    fun=0.0,
                    message=(
                        "both sides required but "
                        "candidate side missing"
                    ),
                    n_candidates=len(candidates),
                    n_selected=0,
                ),
            )

        constraints.extend(
            [
                LinearConstraint(
                    buy_mask.reshape(1, -1),
                    lb=1,
                    ub=np.inf,
                ),
                LinearConstraint(
                    sell_mask.reshape(1, -1),
                    lb=1,
                    ub=np.inf,
                ),
            ]
        )

    # 同scoreの決定性を確保する小さなtie-break。
    tie_break = np.arange(
        n,
        0,
        -1,
        dtype=float,
    ) * 1e-12

    objective = -(
        scores + tie_break
    )

    result = milp(
        c=objective,
        constraints=constraints,
        bounds=Bounds(
            lb=np.zeros(n),
            ub=np.ones(n),
        ),
        integrality=np.ones(
            n,
            dtype=int,
        ),
    )

    if not result.success:
        if not risk.allow_optimizer_fallback:
            raise RuntimeError(
                "MILP optimization failed: "
                f"{result.status} "
                f"{result.message}"
            )

        selected, additional_rejected = (
            _greedy_select(
                valid,
                risk,
                existing_long=existing_long,
                existing_short=existing_short,
                gross_limit_override=(
                    gross_limit_override
                ),
            )
        )

        return (
            selected,
            rejected + additional_rejected,
            OptimizationDiagnostics(
                status=int(result.status),
                fun=0.0,
                message=(
                    "MILP failed; greedy fallback"
                ),
                n_candidates=len(candidates),
                n_selected=len(selected),
            ),
        )

    selected_indices = set(
        np.where(result.x > 0.5)[0]
        .astype(int)
        .tolist()
    )

    selected_orders: list[dict] = []

    for index, row in enumerate(valid):
        clean = {
            key: value
            for key, value in row.items()
            if key != "_risk_value"
        }

        if index in selected_indices:
            selected_orders.append(clean)
        else:
            clean["reason"] = (
                "OPTIMIZER_NOT_SELECTED"
            )
            rejected.append(clean)

    return (
        selected_orders,
        rejected,
        OptimizationDiagnostics(
            status=int(result.status),
            fun=float(result.fun),
            message=str(result.message),
            n_candidates=len(candidates),
            n_selected=len(selected_orders),
            details={
                "selector": "milp",
                "n_constraints": len(
                    constraints
                ),
            },
        ),
    )


def _greedy_select(
    valid: list[dict],
    risk: RiskConfig,
    *,
    existing_long: float,
    existing_short: float,
    gross_limit_override: float | None,
) -> tuple[list[dict], list[dict]]:
    gross_limit = float(
        risk.max_gross_exposure_yen
        if gross_limit_override is None
        else min(
            float(risk.max_gross_exposure_yen),
            float(gross_limit_override),
        )
    )

    buys = sorted(
        [
            row
            for row in valid
            if row["side"] == "BUY"
        ],
        key=lambda row: float(
            row.get("score", 0.0)
        ),
        reverse=True,
    )
    sells = sorted(
        [
            row
            for row in valid
            if row["side"] == "SELL"
        ],
        key=lambda row: float(
            row.get("score", 0.0)
        ),
        reverse=True,
    )

    ordered: list[dict] = []

    for index in range(
        max(len(buys), len(sells))
    ):
        pair = []

        if index < len(buys):
            pair.append(buys[index])

        if index < len(sells):
            pair.append(sells[index])

        pair.sort(
            key=lambda row: float(
                row.get("score", 0.0)
            ),
            reverse=True,
        )
        ordered.extend(pair)

    selected: list[dict] = []
    rejected: list[dict] = []

    long_value = existing_long
    short_value = existing_short

    for row in ordered:
        value = float(
            row["_risk_value"]
        )

        reason = None

        if (
            len(selected)
            >= risk.max_orders_per_day
        ):
            reason = "DAILY_ORDER_LIMIT"
        else:
            next_long = long_value
            next_short = short_value

            if row["side"] == "BUY":
                next_long += value
            else:
                next_short += value

            if (
                next_long
                > risk.max_long_exposure_yen
            ):
                reason = "LONG_LIMIT"
            elif (
                next_short
                > risk.max_short_exposure_yen
            ):
                reason = "SHORT_LIMIT"
            elif (
                next_long + next_short
                > gross_limit
            ):
                reason = "GROSS_LIMIT"

        if reason is not None:
            rejected_row = dict(row)
            rejected_row["reason"] = reason
            rejected.append(rejected_row)
            continue

        long_value = next_long
        short_value = next_short
        selected.append(row)

    while selected:
        net = long_value - short_value

        if (
            abs(net)
            <= risk.max_net_exposure_yen
        ):
            break

        excessive_side = (
            "BUY" if net > 0 else "SELL"
        )

        removable = [
            row
            for row in selected
            if row["side"] == excessive_side
        ]

        if not removable:
            break

        weakest = min(
            removable,
            key=lambda row: float(
                row.get("score", 0.0)
            ),
        )

        selected.remove(weakest)

        if weakest["side"] == "BUY":
            long_value -= float(
                weakest["_risk_value"]
            )
        else:
            short_value -= float(
                weakest["_risk_value"]
            )

        rejected_row = dict(weakest)
        rejected_row["reason"] = "NET_LIMIT"
        rejected.append(rejected_row)

    if (
        abs(long_value - short_value)
        > risk.max_net_exposure_yen
    ):
        for row in selected:
            rejected_row = dict(row)
            rejected_row["reason"] = (
                "NET_LIMIT_UNRESOLVED"
            )
            rejected.append(rejected_row)

        selected = []

    if risk.require_both_sides and selected:
        sides = {
            row["side"]
            for row in selected
        }

        if sides != {"BUY", "SELL"}:
            for row in selected:
                rejected_row = dict(row)
                rejected_row["reason"] = (
                    "REQUIRE_BOTH_SIDES"
                )
                rejected.append(rejected_row)

            selected = []

    def clean(
        rows: list[dict],
    ) -> list[dict]:
        return [
            {
                key: value
                for key, value in row.items()
                if key != "_risk_value"
            }
            for row in rows
        ]

    return clean(selected), clean(rejected)
