"""リスク制限（FR-RISK-01/02/03）。
スケルトン — Part 2 で完全実装。
"""

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
        self.max_orders_per_day = max_orders_per_day
        self.max_gross_exposure_yen = max_gross_exposure_yen
        self.max_single_name_exposure_yen = max_single_name_exposure_yen
        self.max_long_exposure_yen = max_long_exposure_yen
        self.max_short_exposure_yen = max_short_exposure_yen
        self.allow_short_without_confirmed_shortability = (
            allow_short_without_confirmed_shortability
        )


def apply_order_risk_limits(
    orders: pd.DataFrame, risk: RiskConfig
) -> pd.DataFrame:
    """Apply risk limits and return filtered orders.

    TODO: full implementation in Part 2.
    """
    if orders is None or orders.empty:
        return pd.DataFrame()

    out = orders.copy()

    # Filter by shortability
    if not risk.allow_short_without_confirmed_shortability:
        out = out[~((out["side"] == "SELL") & (~out["shortable"]))]

    # Filter by single name exposure
    out = out[out["value_yen"] <= risk.max_single_name_exposure_yen]

    # Limit number of orders per day
    out = out.head(risk.max_orders_per_day)

    return out.reset_index(drop=True)
