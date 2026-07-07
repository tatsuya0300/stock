"""compute_size の cap ロジックと寄成警告の回帰テスト。"""

from jp_signal.sizing import compute_size


def test_cap_limits_target_notional():
    """target_notional が cap を超えると cap でクリップされる。"""
    prev_turnover = 10_000_000.0  # 前日代金
    ref_price = 1000.0
    adv_ratio = 0.001            # 目標 0.1% = 10,000円
    adv_ratio_cap = 0.002        # 上限 0.2% = 20,000円
    # 目標を上限超の 50,000円 指定 → 20,000円にクリップ → 20 株 → 単元100で0株
    qty, yen, _ = compute_size(
        prev_turnover, ref_price, adv_ratio, adv_ratio_cap,
        target_notional=50_000.0, unit=100,
    )
    assert yen <= prev_turnover * adv_ratio_cap


def test_default_uses_adv_ratio():
    """target_notional 未指定なら turnover*adv_ratio を目標にする。"""
    prev_turnover = 100_000_000.0
    ref_price = 1000.0
    # 目標 0.1% = 100,000円 → 100株
    qty, yen, _ = compute_size(prev_turnover, ref_price, 0.001, 0.002, unit=100)
    assert qty == 100
    assert yen == 100_000.0


def test_market_open_unit_cap_warns():
    """寄成で単元上限を超えると警告文字列を返す。"""
    prev_turnover = 1_000_000_000.0
    ref_price = 100.0
    # 目標 0.1% = 1,000,000円 → 10,000株 = 100単元 > cap(50)
    _, _, warn = compute_size(
        prev_turnover, ref_price, 0.001, 0.01, unit=100,
        market_open_unit_cap=50, is_market_open_order=True,
    )
    assert warn != ""


def test_no_warn_when_not_market_open():
    """寄成以外（指値など）では単元上限警告を出さない。"""
    prev_turnover = 1_000_000_000.0
    ref_price = 100.0
    _, _, warn = compute_size(
        prev_turnover, ref_price, 0.001, 0.01, unit=100,
        market_open_unit_cap=50, is_market_open_order=False,
    )
    assert warn == ""


def test_zero_on_invalid_price():
    """参照価格が0以下なら 0 株を返す。"""
    qty, yen, warn = compute_size(1_000_000.0, 0.0, 0.001, 0.002)
    assert qty == 0 and yen == 0.0 and warn == ""
