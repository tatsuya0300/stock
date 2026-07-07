"""執行サイズ算定（FR-SIZE-01/02）。

推奨発注額 = 前日代金 * adv_ratio（上限 adv_ratio_cap）。
株数は単元(unit)単位に切り下げ。寄成 short_unit_cap 単元超は警告文字列を返す。
"""

from __future__ import annotations


def compute_size(
    prev_turnover: float,
    ref_price: float,
    adv_ratio: float,
    adv_ratio_cap: float,
    unit: int = 100,
    short_unit_cap: int = 50,
) -> tuple[int, float, str]:
    """推奨発注サイズを返す。

    returns: (qty, order_value_yen, warn_message)
    """
    if ref_price is None or ref_price <= 0 or prev_turnover is None or prev_turnover <= 0:
        return 0, 0.0, ""

    target_yen = min(prev_turnover * adv_ratio, prev_turnover * adv_ratio_cap)
    raw_qty = target_yen / ref_price
    qty = int(raw_qty // unit) * unit

    warn = ""
    if unit > 0 and (qty // unit) > short_unit_cap:
        warn = f"寄成{short_unit_cap}単元超過: 分割不可、指値必須"

    return qty, qty * ref_price, warn
