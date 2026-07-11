"""非有限値・不正日付のfail-closed動作を検証するテスト。"""

import numpy as np
import pandas as pd
import pytest

from jp_signal.model import MeanReversionRule
from jp_signal.order_builder import signals_to_orders
from jp_signal.risk import (
    RiskConfig,
    select_orders_with_reasons,
)


def test_risk_rejects_nan_risk_value_yen() -> None:
    """risk_value_yen=NaNがSINGLE_NAME_LIMITをすり抜けずINVALID_RISK_VALUE_YENで拒否される。"""
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "value_yen": 1_000_000,
                "risk_value_yen": np.nan,
                "shortable": True,
                "score": 1.0,
            }
        ]
    )

    risk = RiskConfig(
        require_both_sides=False,
        max_net_exposure_yen=100_000_000,
    )

    result = select_orders_with_reasons(
        orders,
        risk,
    )

    assert result.selected.empty
    assert len(result.rejected) == 1
    assert (
        result.rejected.iloc[0]["reason"]
        == "INVALID_RISK_VALUE_YEN"
    )


@pytest.mark.parametrize(
    "risk_value",
    [
        np.inf,
        -np.inf,
        -1.0,
        0.0,
    ],
)
def test_risk_rejects_invalid_risk_value(
    risk_value: float,
) -> None:
    """inf/-inf/負/ゼロのrisk_value_yenが拒否される。"""
    orders = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "value_yen": 1_000_000,
                "risk_value_yen": risk_value,
                "shortable": True,
                "score": 1.0,
            }
        ]
    )

    result = select_orders_with_reasons(
        orders,
        RiskConfig(
            require_both_sides=False,
            max_net_exposure_yen=100_000_000,
        ),
    )

    assert result.selected.empty
    assert (
        result.rejected.iloc[0]["reason"]
        == "INVALID_RISK_VALUE_YEN"
    )


@pytest.mark.parametrize(
    "invalid_limit",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_risk_config_rejects_non_finite_limits(
    invalid_limit: float,
) -> None:
    """RiskConfigの制限値に非有限値を渡すとValueError。"""
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        RiskConfig(
            max_gross_exposure_yen=invalid_limit,
        )


def test_model_accepts_timestamp_dates() -> None:
    """MeanReversionRule.generate()がTimestamp日付でも動作する。"""
    prices = pd.DataFrame(
        [
            ("A", pd.Timestamp("2024-01-04"), 100.0),
            ("A", pd.Timestamp("2024-01-05"), 90.0),
            ("B", pd.Timestamp("2024-01-04"), 100.0),
            ("B", pd.Timestamp("2024-01-05"), 110.0),
        ],
        columns=[
            "code",
            "date",
            "adj_close",
        ],
    )

    model = MeanReversionRule(
        lookback=1,
        top_n=1,
    )

    result = model.generate(
        prices,
        as_of="2024-01-09",
    )

    assert not result.empty

    sides = dict(
        zip(
            result["code"],
            result["side"],
            strict=False,
        )
    )

    assert sides["A"] == "BUY"
    assert sides["B"] == "SELL"


def test_order_builder_accepts_timestamp_dates() -> None:
    """signals_to_orders()がTimestamp日付でも動作する。"""
    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": pd.Timestamp("2024-01-09"),
                "close": 100.0,
                "turnover": 100_000_000.0,
            },
            {
                "code": "7203",
                "date": pd.Timestamp("2024-01-10"),
                "close": 105.0,
                "turnover": 100_000_000.0,
            },
        ]
    )

    signals = pd.DataFrame(
        [
            {
                "code": "7203",
                "side": "BUY",
                "score": 1.0,
                "limit_price": None,
            }
        ]
    )

    orders = signals_to_orders(
        signals,
        prices,
        as_of="2024-01-11",
        sizing_cfg={
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "adv_window": 20,
            "min_adv_periods": 1,
            "require_full_adv_history": False,
            "allow_single_day_turnover_fallback": True,
        },
        risk_cfg=RiskConfig(
            require_both_sides=False,
            max_net_exposure_yen=100_000_000,
        ),
    )

    assert len(orders) == 1
    assert orders.iloc[0]["code"] == "7203"
