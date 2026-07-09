"""執行サイズ算定（FR-SIZE-01/02）。改訂版。

修正点:
  - target_notional（目標発注額）を引数化し、adv_ratio_cap を真の上限として機能させる。
    改訂前は min(turnover*adv_ratio, turnover*adv_ratio_cap) で常に小さい方=adv_ratio
    が選ばれ、cap が意味を持っていなかった。
  - short_unit_cap を market_open_unit_cap に改名し、寄成注文のときのみ警告する
    （売り固有の制約と寄成の単元上限を分離）。

推奨発注額 = target_notional（未指定なら 前日代金 * adv_ratio）。ただし
前日代金 * adv_ratio_cap を上限とする。株数は単元(unit)単位に切り下げ。
"""

from __future__ import annotations


def compute_size(
    prev_turnover: float,
    ref_price: float,
    adv_ratio: float,
    adv_ratio_cap: float,
    target_notional: float | None = None,
    unit: int = 100,
    market_open_unit_cap: int = 50,
    is_market_open_order: bool = True,
    enforce_market_open_unit_cap: bool = False,
) -> tuple[int, float, str]:
    """推奨発注サイズを返す。

    Args:
        prev_turnover: 前日売買代金（円、ADV近似）。
        ref_price: 参照価格（前日終値など）。
        adv_ratio: 目標額が未指定のときのデフォルト発注比率（例 0.001 = 前日代金の0.1%）。
        adv_ratio_cap: 前日代金に対する発注額の上限比率（例 0.002 = 0.2%）。
        target_notional: 上位ロジックが決めた目標発注額（円）。None なら turnover*adv_ratio。
        unit: 単元株数。0 は拒否。
        market_open_unit_cap: 寄成で許容する最大単元数。
        is_market_open_order: 寄成注文か（True のときのみ寄成上限を警告）。
        enforce_market_open_unit_cap: True の場合、寄成の単元上限を超える分をクリップする。

    Returns:
        (qty, order_value_yen, warn_message)

    Raises:
        ValueError: adv_ratio > adv_ratio_cap の場合、または
                    target_notional < 0 の場合、または unit <= 0 の場合。
    """
    # バリデーション
    if adv_ratio > adv_ratio_cap:
        raise ValueError(
            f"adv_ratio ({adv_ratio}) が adv_ratio_cap ({adv_ratio_cap}) を超えています"
        )
    if target_notional is not None and target_notional < 0:
        raise ValueError(f"target_notional ({target_notional}) は負の値にできません")
    if unit <= 0:
        raise ValueError(f"unit ({unit}) は正の整数である必要があります")

    if (
        ref_price is None
        or ref_price <= 0
        or prev_turnover is None
        or prev_turnover <= 0
    ):
        return 0, 0.0, ""

    # 目標額（未指定は turnover*adv_ratio）を、turnover*adv_ratio_cap で上限クリップ。
    desired = target_notional if target_notional is not None else prev_turnover * adv_ratio
    cap_yen = prev_turnover * adv_ratio_cap
    target_yen = min(desired, cap_yen)

    raw_qty = target_yen / ref_price
    qty = int(raw_qty // unit) * unit

    warn = ""
    if is_market_open_order and unit > 0 and (qty // unit) > market_open_unit_cap:
        if enforce_market_open_unit_cap:
            qty = market_open_unit_cap * unit
            warn = f"寄成{market_open_unit_cap}単元超過: {qty}株にクリップ"
        else:
            warn = f"寄成{market_open_unit_cap}単元超過: 分割 or 指値を検討"

    return qty, qty * ref_price, warn
