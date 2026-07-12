from datetime import date

import pandas as pd

from jp_signal.historical_orders import (
    generate_historical_orders,
)


class DummyModel:
    def generate(
        self,
        prices: pd.DataFrame,
        as_of: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "code": "7203",
                    "side": "BUY",
                    "score": 1.0,
                    "limit_price": None,
                }
            ]
        )


def _price_loader(
    codes: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        end=pd.Timestamp(end) - pd.Timedelta(days=1),
        periods=30,
    )

    rows = []

    for code in codes:
        for d in dates:
            rows.append(
                {
                    "code": code,
                    "date": d.strftime("%Y-%m-%d"),
                    "open": 1000.0,
                    "high": 1010.0,
                    "low": 990.0,
                    "close": 1000.0,
                    "adj_open": 1000.0,
                    "adj_high": 1010.0,
                    "adj_low": 990.0,
                    "adj_close": 1000.0,
                    "volume": 1_000_000.0,
                    "turnover": 1_000_000_000.0,
                }
            )

    return pd.DataFrame(rows)


def _universe_loader(
    as_of: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "7203",
                "name": "Toyota",
            }
        ]
    )


def test_generate_historical_orders_builds_orders():
    result = generate_historical_orders(
        trading_dates=[
            date(2026, 7, 10),
        ],
        model_factory=lambda cfg: DummyModel(),
        price_loader=_price_loader,
        shortability_loader=None,
        universe_loader=_universe_loader,
        model_cfg={
            "lookback": 5,
        },
        sizing_cfg={
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "adv_window": 20,
            "min_adv_periods": 20,
            "require_full_adv_history": True,
            "allow_single_day_turnover_fallback": False,
            "market_open_unit_cap": 50,
        },
        risk_cfg={
            "max_orders_per_day": 10,
            "max_gross_exposure_yen": 100_000_000,
            "max_single_name_exposure_yen": 20_000_000,
            "max_long_exposure_yen": 50_000_000,
            "max_short_exposure_yen": 50_000_000,
            "max_net_exposure_yen": 50_000_000,
            "require_both_sides": False,
            "allow_short_without_confirmed_shortability": False,
        },
        holding_days=1,
        unit=100,
    )

    assert len(result.orders) == 1

    order = result.orders.iloc[0]

    assert order["date"] == "2026-07-10"
    assert order["code"] == "7203"
    assert order["side"] == "BUY"
    assert order["qty"] == 1000
    assert order["holding_days"] == 1

    assert result.diagnostics["signals_generated"] == 1
    assert result.diagnostics["orders_produced"] == 1


def test_generate_historical_orders_fails_fast():
    def broken_loader(
        codes: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        raise RuntimeError("price loader failed")

    try:
        generate_historical_orders(
            trading_dates=[
                date(2026, 7, 10),
            ],
            model_factory=lambda cfg: DummyModel(),
            price_loader=broken_loader,
            universe_loader=_universe_loader,
            model_cfg={"lookback": 5},
            sizing_cfg={
                "adv_ratio": 0.001,
                "adv_ratio_cap": 0.002,
            },
            risk_cfg={
                "require_both_sides": False,
            },
            fail_fast=True,
        )
    except RuntimeError as exc:
        assert str(exc) == "price loader failed"
    else:
        raise AssertionError(
            "expected RuntimeError"
        )
