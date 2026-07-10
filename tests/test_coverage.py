import pandas as pd

from jp_signal.coverage import CoverageThresholds, validate_daily_coverage


def test_validate_daily_coverage_passes_complete_data():
    universe = pd.DataFrame(
        [
            {"code": "7203", "name": "Toyota"},
            {"code": "6758", "name": "Sony"},
        ]
    )

    rows = []
    for code in ["7203", "6758"]:
        for d in ["2024-01-05", "2024-01-09", "2024-01-10"]:
            rows.append(
                {
                    "code": code,
                    "date": d,
                    "open": 100,
                    "high": 110,
                    "low": 90,
                    "close": 100,
                    "turnover": 1_000_000,
                }
            )

    prices = pd.DataFrame(rows)

    report = validate_daily_coverage(
        prices,
        universe,
        as_of="2024-01-11",
        lookback=1,
        adv_window=2,
        min_adv_periods=2,
        thresholds=CoverageThresholds(
            price_coverage_min=1.0,
            lookback_coverage_min=1.0,
            turnover_coverage_min=1.0,
        ),
    )

    assert report.ok
    assert report.price_coverage == 1.0
    assert report.lookback_coverage == 1.0
    assert report.turnover_coverage == 1.0


def test_validate_daily_coverage_fails_partial_data():
    universe = pd.DataFrame(
        [
            {"code": "7203", "name": "Toyota"},
            {"code": "6758", "name": "Sony"},
        ]
    )

    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-10",
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 100,
                "turnover": 1_000_000,
            }
        ]
    )

    report = validate_daily_coverage(
        prices,
        universe,
        as_of="2024-01-11",
        lookback=1,
        adv_window=2,
        min_adv_periods=1,
        thresholds=CoverageThresholds(
            price_coverage_min=1.0,
            lookback_coverage_min=1.0,
            turnover_coverage_min=1.0,
        ),
    )

    assert not report.ok
    assert "PRICE_COVERAGE_LOW" in report.failed_reasons
