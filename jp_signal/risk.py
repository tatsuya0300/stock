"""リスク制限（FR-RISK-01/02/03）。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd


@dataclass(frozen=True)
class RiskSelectionResult:
    """リスク選択結果。

    selected:
        採用された注文。

    rejected:
        不採用注文。元の注文列にreasonを追加する。
    """

    selected: pd.DataFrame
    rejected: pd.DataFrame


class RiskConfig:
    """Order-level risk limits."""

    def __init__(
        self,
        max_orders_per_day: int = 10,
        max_gross_exposure_yen: float = 100_000_000.0,
        max_single_name_exposure_yen: float = 20_000_000.0,
        max_long_exposure_yen: float = 50_000_000.0,
        max_short_exposure_yen: float = 50_000_000.0,
        max_net_exposure_yen: float = 5_000_000.0,
        require_both_sides: bool = True,
        allow_short_without_confirmed_shortability: bool = False,
    ):
        if max_orders_per_day < 1:
            raise ValueError(f"max_orders_per_day must be >= 1: {max_orders_per_day}")

        non_negative_limits = {
            "max_gross_exposure_yen": max_gross_exposure_yen,
            "max_single_name_exposure_yen": max_single_name_exposure_yen,
            "max_long_exposure_yen": max_long_exposure_yen,
            "max_short_exposure_yen": max_short_exposure_yen,
            "max_net_exposure_yen": max_net_exposure_yen,
        }

        for name, value in non_negative_limits.items():
            if float(value) < 0:
                raise ValueError(f"{name} must be >= 0: {value}")

        self.max_orders_per_day = int(max_orders_per_day)
        self.max_gross_exposure_yen = float(max_gross_exposure_yen)
        self.max_single_name_exposure_yen = float(max_single_name_exposure_yen)
        self.max_long_exposure_yen = float(max_long_exposure_yen)
        self.max_short_exposure_yen = float(max_short_exposure_yen)
        self.max_net_exposure_yen = float(max_net_exposure_yen)
        self.require_both_sides = bool(require_both_sides)
        self.allow_short_without_confirmed_shortability = bool(
            allow_short_without_confirmed_shortability
        )


def risk_config_from_dict(d: dict) -> RiskConfig:
    """設定 dict から RiskConfig を生成する。"""
    return RiskConfig(
        max_orders_per_day=int(d.get("max_orders_per_day", 10)),
        max_gross_exposure_yen=float(d.get("max_gross_exposure_yen", 100_000_000.0)),
        max_single_name_exposure_yen=float(d.get("max_single_name_exposure_yen", 20_000_000.0)),
        max_long_exposure_yen=float(d.get("max_long_exposure_yen", 50_000_000.0)),
        max_short_exposure_yen=float(d.get("max_short_exposure_yen", 50_000_000.0)),
        max_net_exposure_yen=float(d.get("max_net_exposure_yen", 5_000_000.0)),
        require_both_sides=bool(d.get("require_both_sides", True)),
        allow_short_without_confirmed_shortability=bool(
            d.get(
                "allow_short_without_confirmed_shortability",
                False,
            )
        ),
    )


def select_orders_with_reasons(
    orders: pd.DataFrame,
    risk: RiskConfig,
    score_col: str = "score",
) -> RiskSelectionResult:
    """注文を選択し、全ての不採用理由を返す。

    require_both_sides=True の場合は、単純な全体score順ではなく、
    BUY/SELLを交互に評価することで片側だけが注文枠を消費することを防ぐ。

    注意:
        厳密な数理最適化ではない。決定論的なgreedy heuristicである。
    """
    if orders is None or orders.empty:
        empty = pd.DataFrame()
        return RiskSelectionResult(
            selected=empty,
            rejected=empty,
        )

    out = orders.copy().reset_index(drop=True)
    out["_source_order"] = range(len(out))

    if "side" not in out.columns:
        raise ValueError("orders missing column: side")
    if "value_yen" not in out.columns:
        raise ValueError("orders missing column: value_yen")

    out["side"] = out["side"].astype(str).str.upper()
    out["value_yen"] = pd.to_numeric(
        out["value_yen"],
        errors="coerce",
    ).fillna(0.0)

    if score_col not in out.columns:
        out[score_col] = 0.0

    out[score_col] = pd.to_numeric(
        out[score_col],
        errors="coerce",
    ).fillna(0.0)

    if "shortable" not in out.columns:
        out["shortable"] = False

    out["shortable"] = out["shortable"].fillna(False).astype(bool)

    rejected_rows: list[dict] = []
    valid_rows: list[dict] = []

    def reject(row: pd.Series | dict, reason: str) -> None:
        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        data["reason"] = reason
        rejected_rows.append(data)

    # 単注文単位で判定できる制約を先に処理する。
    seen_codes: set[str] = set()

    sorted_input = out.sort_values(
        [score_col, "_source_order"],
        ascending=[False, True],
        kind="stable",
    )

    for _, row in sorted_input.iterrows():
        code = str(row.get("code", "")).strip()
        side = str(row["side"])
        value = float(row["value_yen"])
        risk_value = float(row.get("risk_value_yen", value))

        if not code:
            reject(row, "EMPTY_CODE")
            continue

        if code in seen_codes:
            reject(row, "DUPLICATE_CODE")
            continue

        if side not in {"BUY", "SELL"}:
            reject(row, "INVALID_SIDE")
            continue

        if not pd.notna(value) or value <= 0:
            reject(row, "INVALID_VALUE_YEN")
            continue

        if risk_value > risk.max_single_name_exposure_yen:
            reject(row, "SINGLE_NAME_LIMIT")
            continue

        if (
            side == "SELL"
            and not risk.allow_short_without_confirmed_shortability
            and not bool(row["shortable"])
        ):
            reject(row, "NOT_SHORTABLE")
            continue

        seen_codes.add(code)
        valid_rows.append(row.to_dict())

    if not valid_rows:
        selected = out.iloc[0:0].drop(
            columns=["_source_order"],
            errors="ignore",
        )
        rejected = pd.DataFrame(rejected_rows).drop(
            columns=["_source_order"],
            errors="ignore",
        )
        return RiskSelectionResult(selected, rejected)

    valid = pd.DataFrame(valid_rows)

    buys = (
        valid[valid["side"] == "BUY"]
        .sort_values(
            [score_col, "_source_order"],
            ascending=[False, True],
            kind="stable",
        )
        .to_dict("records")
    )
    sells = (
        valid[valid["side"] == "SELL"]
        .sort_values(
            [score_col, "_source_order"],
            ascending=[False, True],
            kind="stable",
        )
        .to_dict("records")
    )

    # require_both_sidesの場合、BUY/SELLを交互に並べる。
    if risk.require_both_sides:
        candidate_order: list[dict] = []
        max_length = max(len(buys), len(sells))

        for i in range(max_length):
            pair: list[dict] = []

            if i < len(buys):
                pair.append(buys[i])
            if i < len(sells):
                pair.append(sells[i])

            # 同一ペア内ではscore順
            pair.sort(
                key=lambda x: (
                    -float(x.get(score_col, 0.0)),
                    int(x["_source_order"]),
                ),
            )
            candidate_order.extend(pair)
    else:
        candidate_order = []
        bi, si = 0, 0

        while bi < len(buys) or si < len(sells):
            if bi < len(buys):
                candidate_order.append(buys[bi])
                bi += 1
            if si < len(sells):
                candidate_order.append(sells[si])
                si += 1

    # 一括制約を逐次評価
    selected_rows: list[dict] = []
    long_value = 0.0
    short_value = 0.0

    for row in candidate_order:
        code = str(row["code"])
        side = str(row["side"])
        value = float(row["value_yen"])
        risk_value = float(row.get("risk_value_yen", value))

        if len(selected_rows) >= risk.max_orders_per_day:
            reject(row, "DAILY_ORDER_LIMIT")
            continue

        next_long = long_value
        next_short = short_value

        if side == "BUY":
            next_long += risk_value
        else:
            next_short += risk_value

        next_gross = next_long + next_short

        if next_long > risk.max_long_exposure_yen:
            reject(row, "LONG_LIMIT")
            continue

        if next_short > risk.max_short_exposure_yen:
            reject(row, "SHORT_LIMIT")
            continue

        if next_gross > risk.max_gross_exposure_yen:
            reject(row, "GROSS_LIMIT")
            continue

        selected_rows.append(row)
        long_value = next_long
        short_value = next_short

    # net exposure制約を満たすまで過剰側の低score注文を除く。
    while selected_rows:
        net_value = long_value - short_value

        if abs(net_value) <= risk.max_net_exposure_yen:
            break

        excessive_side = "BUY" if net_value > 0 else "SELL"

        removable = [row for row in selected_rows if row["side"] == excessive_side]

        if not removable:
            break

        weakest = min(
            removable,
            key=lambda x: (
                float(x.get(score_col, 0.0)),
                -int(x["_source_order"]),
            ),
        )

        selected_rows.remove(weakest)
        risk_value = float(weakest.get("risk_value_yen", weakest["value_yen"]))

        if weakest["side"] == "BUY":
            long_value -= risk_value
        else:
            short_value -= risk_value

        reject(weakest, "NET_LIMIT")

    final_net = long_value - short_value

    if abs(final_net) > risk.max_net_exposure_yen:
        for row in selected_rows:
            reject(row, "NET_LIMIT_UNRESOLVED")
        selected_rows = []

    if risk.require_both_sides and selected_rows:
        selected_sides = {str(row["side"]) for row in selected_rows}

        if not {"BUY", "SELL"}.issubset(selected_sides):
            for row in selected_rows:
                reject(row, "REQUIRE_BOTH_SIDES")
            selected_rows = []

    selected = pd.DataFrame(selected_rows) if selected_rows else valid.iloc[0:0].copy()

    rejected = pd.DataFrame(rejected_rows) if rejected_rows else pd.DataFrame()

    selected = selected.drop(
        columns=["_source_order"],
        errors="ignore",
    ).reset_index(drop=True)

    rejected = rejected.drop(
        columns=["_source_order"],
        errors="ignore",
    ).reset_index(drop=True)

    return RiskSelectionResult(
        selected=selected,
        rejected=rejected,
    )


def apply_order_risk_limits(
    orders: pd.DataFrame,
    risk: RiskConfig,
    score_col: str = "score",
) -> pd.DataFrame:
    """後方互換API。

    不採用理由が必要な呼び出し側は
    select_orders_with_reasons() を使用する。
    """
    return select_orders_with_reasons(
        orders,
        risk,
        score_col=score_col,
    ).selected
return select_orders_with_reasons(
        orders,
        risk,
        score_col=score_col,
    ).selected
