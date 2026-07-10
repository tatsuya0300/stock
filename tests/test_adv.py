import numpy as np
import pandas as pd
import pytest

from jp_signal.adv import rolling_adv_before


def test_rolling_adv_before_no_lookahead():
    prices = pd.DataFrame(
        [
            {"code": "7203", "date": "2024-01-01", "turnover": 100.0},
            {"code": "7203", "date": "2024-01-02", "turnover": 200.0},
            {"code": "7203", "date": "2024-01-03", "turnover": 10000.0},
        ]
    )

    adv = rolling_adv_before(
        prices,
        "2024-01-03",
        window=2,
        min_periods=2,
        strictly_before=True,
    )

    assert float(adv["7203"]) == 150.0


def test_rolling_adv_before_insufficient_history_is_nan():
    prices = pd.DataFrame(
        [
            {"code": "7203", "date": "2024-01-01", "turnover": 100.0},
        ]
    )

    adv = rolling_adv_before(
        prices,
        "2024-01-02",
        window=2,
        min_periods=2,
        strictly_before=True,
    )

    assert np.isnan(float(adv["7203"]))


def test_rolling_adv_before_rejects_invalid_window():
    prices = pd.DataFrame(
        [{"code": "7203", "date": "2024-01-01", "turnover": 100.0}]
    )

    with pytest.raises(ValueError):
        rolling_adv_before(
            prices,
            "2024-01-02",
            window=0,
            min_periods=1,
        )
