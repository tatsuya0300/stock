"""PR-3: Model and risk consistency tests."""
import pandas as pd
import pytest

from jp_signal.model import (
    MeanReversionConfig,
    VolatilityAdjustedMeanReversion,
    model_from_config,
)
from jp_signal.risk import (
    RiskConfig,
    select_orders_with_reasons,
)


def test_model_factory_rejects_unknown_model():
    with pytest.raises(ValueError, match="unsupported model.type"):
        model_from_config({"type": "unknown"})


def test_volatility_model_enforces_top_n_per_side():
    model = VolatilityAdjustedMeanReversion(
        MeanReversionConfig(lookback=2, vol_lookback=2, z_entry=0.0, top_n=2)
    )

    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    rows = []
    for index, code in enumerate(["1001", "1002", "1003", "1004", "1005", "1006"]):
        for i, day in enumerate(dates):
            price = 100.0 + i * (index + 1)
            rows.append({"code": code, "date": day, "close": price, "adj_close": price})

    result = model.generate(pd.DataFrame(rows), as_of="2026-01-20")
    assert len(result[result["side"] == "BUY"]) <= 2
    assert len(result[result["side"] == "SELL"]) <= 2


def test_risk_config_rejects_milp():
    with pytest.raises(NotImplementedError, match="selection_method='milp'"):
        RiskConfig(selection_method="milp")


def test_risk_config_rejects_invalid_selection_method():
    with pytest.raises(ValueError, match="selection_method must be"):
        RiskConfig(selection_method="invalid")


def test_risk_config_defaults_to_greedy():
    config = RiskConfig()
    assert config.selection_method == "greedy"


def test_select_orders_rejects_milp():
    risk = RiskConfig(selection_method="greedy")
    orders = pd.DataFrame()
    with pytest.raises(NotImplementedError, match="selection_method='milp'"):
        # Create a risk config with milp
        risk_milp = RiskConfig.__new__(RiskConfig)
        risk_milp.selection_method = "milp"
        # Need to set other attributes to avoid AttributeError
        for attr in ['max_orders_per_day', 'max_gross_exposure_yen',
                     'max_single_name_exposure_yen', 'max_long_exposure_yen',
                     'max_short_exposure_yen', 'max_net_exposure_yen',
                     'require_both_sides', 'allow_short_without_confirmed_shortability']:
            setattr(risk_milp, attr, getattr(risk, attr))
        result = select_orders_with_reasons(orders, risk_milp)


def test_mean_reversion_config_validates_lookback():
    with pytest.raises(ValueError, match="lookback must be >= 2"):
        MeanReversionConfig(lookback=1, vol_lookback=2, top_n=1)


def test_mean_reversion_config_validates_vol_lookback():
    with pytest.raises(ValueError, match="vol_lookback must be >= 2"):
        MeanReversionConfig(lookback=2, vol_lookback=1, top_n=1)


def test_risk_config_has_no_allow_optimizer_fallback():
    config = RiskConfig()
    assert not hasattr(config, "allow_optimizer_fallback")


def test_risk_config_from_dict_no_allow_optimizer_fallback():
    from jp_signal.risk import risk_config_from_dict
    config = risk_config_from_dict({"allow_optimizer_fallback": True})
    assert not hasattr(config, "allow_optimizer_fallback")
