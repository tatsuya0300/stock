"""東証営業日カレンダー（NFR-OPS）。

土日・祝日（jpholiday）・年末年始（12/31, 1/1-1/3）を非営業日として扱う。
"""

from __future__ import annotations

from datetime import date, timedelta

import jpholiday

# 年末年始の休場日（東証）
_YEAR_END_HOLIDAYS = {(12, 31), (1, 1), (1, 2), (1, 3)}


def is_tse_business_day(d: date) -> bool:
    """東証の営業日判定。土日・祝日・年末年始を除く。"""
    if d.weekday() >= 5:  # 5=土, 6=日
        return False
    if jpholiday.is_holiday(d):
        return False
    if (d.month, d.day) in _YEAR_END_HOLIDAYS:
        return False
    return True


def previous_business_day(d: date) -> date:
    """d の直前の営業日を返す。"""
    x = d - timedelta(days=1)
    while not is_tse_business_day(x):
        x -= timedelta(days=1)
    return x


def next_business_day(d: date) -> date:
    """d の直後の営業日を返す。"""
    x = d + timedelta(days=1)
    while not is_tse_business_day(x):
        x += timedelta(days=1)
    return x
