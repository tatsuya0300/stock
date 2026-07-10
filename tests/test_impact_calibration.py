from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jp_signal.impact_calibration import (
    calibrate_square_root_impact,
)


def test_recovers_square_root_impact_coefficient():
    rows = []

    for i in range(1, 51):
        expected = 1000.0
        qty = 100.0 * i
        adv = 100_000_000.0

        participation = expected * qty / adv

        slippage_bp = 2.0 + 30.0 * np.sqrt(participation)

        fill_price = expected * (1.0 + slippage_bp / 10_000.0)

        rows.append(
            {
                "expected_price": expected,
                "price": fill_price,
                "qty": qty,
                "adv": adv,
                "side": "BUY",
            }
        )

    result = calibrate_square_root_impact(
        pd.DataFrame(rows),
        minimum_observations=30,
    )

    assert result.intercept_bp == pytest.approx(
        2.0,
        abs=1e-8,
    )
    assert result.impact_k_bp == pytest.approx(
        30.0,
        abs=1e-8,
    )
    assert result.r_squared == pytest.approx(
        1.0,
        abs=1e-8,
    )
