"""coverage判定でmin_adv_periodsを正しく使う回帰テスト。

2024年1月の日本市場: 1/8(月)は成人の日で休場のため、
1/5(金)が1/9(火)の前営業日になる。
"""

from __future__ import annotations

import pandas as pd

from jp_signal.coverage import (
    CoverageThresholds,
    validate_daily_coverage,
)


def test_coverage_uses_min_adv_periods_not_full_adv_window():
    """min_adv_periodsが満たされればadv_window未満でもcoverage OKになる。

    Jan 9(Tue) → previous business day = Jan 5(Fri) [1/8は成人の日で休場]
    Jan 5より前のデータ: Jan 4(Thu)のみ → 1行
    lookback=2, min_adv_periods=2 → required_bars=2
    だがhistoryは1行しかないのでlookback_coverage=0.0になる。

    そこで、history側にもっとデータを追加してlookback=2を満たすようにする。
    """
    prices = pd.DataFrame(
        [
            {
                "code": "7203",
                "date": "2024-01-01",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "turnover": 100_000_000,
            },
            {
                "code": "7203",
                "date": "2024-01-02",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "turnover": 100_000_000,
            },
            {
                # target_date = 2024-01-05 (Fri, 1/9 Tueの前営業日)
                "code": "7203",
                "date": "2024-01-05",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "turnover": 100_000_000,
            },
        ]
    )

    universe = pd.DataFrame([{"code": "7203", "name": "Toyota"}])

    report = validate_daily_coverage(
        prices,
        universe,
        as_of="2024-01-09",
        lookback=2,
        adv_window=20,
        min_adv_periods=2,
        thresholds=CoverageThresholds(
            price_coverage_min=1.0,
            lookback_coverage_min=1.0,
            turnover_coverage_min=1.0,
            hard_fail=True,
        ),
    )

    # required_bars = max(lookback=2, min_adv_periods=2) = 2
    # 1/5より前のデータ: 1/1, 1/2 → 2行 >= 2 → OK
    assert report.ok, f"report={report}"
    assert report.lookback_available_count == 1
    assert report.lookback_coverage == 1.0
