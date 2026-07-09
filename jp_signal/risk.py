"""リスク制限（FR-RISK-01/02/03）。"""

from __future__ import annotations

import pandas as pd


class RiskConfig:
    """Order-level risk limits."""

    def __init__(
        self,
        max_orders_per_day: int = 10,
        max_gross_exposure_yen: float = 100_000_000.0,
        max_single_name_exposure_yen: float = 20_000_000.0,
        max_long_exposure_yen: float = 100_000_000.0,
        max_short_exposure_yen: float = 100_000_000.0,
        allow_short_without_confirmed_shortability: bool = False,
    ):
        self.max_orders_per_day = int(max_orders_per_day)
        self.max_gross_exposure_yen = float(max_gross_exposure_yen)
        self.max_single_name_exposure_yen = float(max_single_name_exposure_yen)
        self.max_long_exposure_yen = float(max_long_exposure_yen)
        self.max_short_exposure_yen = float(max_short_exposure_yen)
        self.allow_short_without_confirmed_shortability = bool(
            allow_short_without_confirmed_shortability
        )


def risk_config_from_dict(d: dict) -> RiskConfig:
    """設定 dict から RiskConfig を生成する。"""
    return RiskConfig(
        max_orders_per_day=int(d.get("max_orders_per_day", 10)),
        max_gross_exposure_yen=float(d.get("max_gross_exposure_yen", 100_000_000.0)),
        max_single_name_exposure_yen=float(
            d.get("max_single_name_exposure_yen", 20_000_000.0)
        ),
        max_long_exposure_yen=float(d.get("max_long_exposure_yen", 100_000_000.0)),
        max_short_exposure_yen=float(d.get("max_short_exposure_yen", 100_000_000.0)),
        allow_short_without_confirmed_shortability=bool(
            d.get("allow_short_without_confirmed_shortability", False)
        ),
    )


def apply_order_risk_limits(
    orders: pd.DataFrame,
    risk: RiskConfig,
    score_col: str = "score",
) -> pd.DataFrame:
    """リスク制限を適用して注文を絞り込む。

    - SELL は shortable 確認済みのみ
    - value_yen が正のもののみ
    - 1銘柄あたり max_single_name_exposure_yen 以下
    - 総エクスポージャー制限（BUY/SELL 別 + グロス）
    - 発注件数上限

    Args:
        orders: code, side, value_yen, shortable を含む DataFrame。
        risk: RiskConfig インスタンス。
        score_col: スコアリング用列（降順ソートに使用）。

    Returns:
        フィルター後の orders。
    """
    if orders is None or orders.empty:
        return pd.DataFrame() if orders is None else orders.iloc[0:0].copy()

    out = orders.copy()
    out["side"] = out["side"].astype(str).str.upper()
    out["value_yen"] = pd.to_numeric(out["value_yen"], errors="coerce").fillna(0.0)

    if "shortable" not in out.columns:
        out["shortable"] = False
    out["shortable"] = out["shortable"].fillna(False).astype(bool)

    # SELL は shortable 確認済みのみ
    if not risk.allow_short_without_confirmed_shortability:
        out = out[~((out["side"] == "SELL") & (~out["shortable"]))]

    out = out[out["value_yen"] > 0]
    out = out[out["value_yen"] <= risk.max_single_name_exposure_yen]

    if out.empty:
        return out

    # スコア降順でソート
    if score_col in out.columns:
        out = out.sort_values(score_col, ascending=False)
    else:
        out = out.sort_values("value_yen", ascending=False)

    selected: list[dict] = []
    gross = 0.0
    long_exp = 0.0
    short_exp = 0.0
    seen_codes: set[str] = set()

    for _, row in out.iterrows():
        code = str(row.get("code", ""))
        if not code:
            continue

        if len(selected) >= risk.max_orders_per_day:
            break

        if code in seen_codes:
            continue

        v = float(row["value_yen"])
        if gross + v > risk.max_gross_exposure_yen:
            continue

        if row["side"] == "BUY":
            if long_exp + v > risk.max_long_exposure_yen:
                continue
            long_exp += v
        else:
            if short_exp + v > risk.max_short_exposure_yen:
                continue
            short_exp += v

        gross += v
        seen_codes.add(code)
        selected.append(row)

    if not selected:
        return out.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)
