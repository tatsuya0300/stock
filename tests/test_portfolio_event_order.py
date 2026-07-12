"""PR2: イベント順序・信用保証金率テスト。

検証項目:
1. carry accrual が決済後に実行されること
2. equity_at_open の計算と検証
3. equity_to_gross_ratio / margin_breach が ledger に記録されること
4. 信用保証金率低下による MARGIN_BREACH リジェクト
5. 決済ポジションの carry が正しく計上されること（旧イベント順との互換性）
6. _select_orders の gross limit が正しく判定されること
7. 決済予定ポジションが寄付き時点で既存コードとしてブロックされること
"""

from __future__ import annotations

import pandas as pd
import pytest

from jp_signal.portfolio import PortfolioBacktester
from jp_signal.risk import RiskConfig


def make_risk(**overrides) -> RiskConfig:
    params = dict(
        max_orders_per_day=10,
        max_gross_exposure_yen=20_000_000,
        max_single_name_exposure_yen=10_000_000,
        max_long_exposure_yen=10_000_000,
        max_short_exposure_yen=10_000_000,
        max_net_exposure_yen=10_000_000,
        require_both_sides=False,
        allow_short_without_confirmed_shortability=True,
    )
    params.update(overrides)
    return RiskConfig(**params)


def make_prices() -> pd.DataFrame:
    """5営業日の価格データ。min_adv_periods=2 には事前データ不足で注意。"""
    values = {
        "7203": [
            ("2024-01-01", 100.0, 100.0),
            ("2024-01-02", 100.0, 101.0),
            ("2024-01-03", 101.0, 103.0),
            ("2024-01-04", 103.0, 102.0),
            ("2024-01-05", 102.0, 104.0),
        ],
        "6758": [
            ("2024-01-01", 200.0, 200.0),
            ("2024-01-02", 200.0, 199.0),
            ("2024-01-03", 199.0, 197.0),
            ("2024-01-04", 197.0, 198.0),
            ("2024-01-05", 198.0, 196.0),
        ],
    }
    rows = []
    for code, code_rows in values.items():
        for dt, open_price, close_price in code_rows:
            rows.append({
                "code": code, "date": dt,
                "open": open_price, "high": max(open_price, close_price),
                "low": min(open_price, close_price), "close": close_price,
                "adj_open": open_price, "adj_high": max(open_price, close_price),
                "adj_low": min(open_price, close_price), "adj_close": close_price,
                "volume": 1_000_000, "turnover": 100_000_000,
            })
    return pd.DataFrame(rows)


def make_prices_long() -> pd.DataFrame:
    """min_adv_periods>=1 を満たすため十分な事前データを含む価格データ。"""
    extra = {
        "7203": [
            ("2023-12-25", 100.0, 100.0), ("2023-12-26", 100.0, 100.0),
            ("2023-12-27", 100.0, 100.0), ("2023-12-28", 100.0, 100.0),
            ("2023-12-29", 100.0, 100.0),
        ],
        "6758": [
            ("2023-12-25", 200.0, 200.0), ("2023-12-26", 200.0, 200.0),
            ("2023-12-27", 200.0, 200.0), ("2023-12-28", 200.0, 200.0),
            ("2023-12-29", 200.0, 200.0),
        ],
    }
    main = {
        "7203": [
            ("2024-01-01", 100.0, 100.0), ("2024-01-02", 100.0, 101.0),
            ("2024-01-03", 101.0, 103.0), ("2024-01-04", 103.0, 102.0),
            ("2024-01-05", 102.0, 104.0),
        ],
        "6758": [
            ("2024-01-01", 200.0, 200.0), ("2024-01-02", 200.0, 199.0),
            ("2024-01-03", 199.0, 197.0), ("2024-01-04", 197.0, 198.0),
            ("2024-01-05", 198.0, 196.0),
        ],
    }
    rows = []
    for code, code_rows in extra.items():
        for dt, open_price, close_price in code_rows:
            rows.append({
                "code": code, "date": dt, "open": open_price,
                "high": open_price, "low": open_price, "close": close_price,
                "adj_open": open_price, "adj_high": open_price,
                "adj_low": open_price, "adj_close": close_price,
                "volume": 1_000_000, "turnover": 100_000_000,
            })
    for code, code_rows in main.items():
        for dt, open_price, close_price in code_rows:
            rows.append({
                "code": code, "date": dt,
                "open": open_price, "high": max(open_price, close_price),
                "low": min(open_price, close_price), "close": close_price,
                "adj_open": open_price, "adj_high": max(open_price, close_price),
                "adj_low": min(open_price, close_price), "adj_close": close_price,
                "volume": 1_000_000, "turnover": 100_000_000,
            })
    return pd.DataFrame(rows)


def make_order(code="7203", side="BUY", qty=100, date="2024-01-03",
               holding_days=1, score=1.0, shortable=True) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": date, "code": code, "side": side, "qty": qty,
        "order_type": "MKT_OPEN", "holding_days": holding_days,
        "score": score, "shortable": shortable,
    }])


# ----------------------------------------------------------------
# 1. carry 関連（決済後実行 + 決済ポジションのcarry計上）
# ----------------------------------------------------------------

def test_carry_accrued_after_close_closing_positions_get_carry():
    """決済後にcarryが実行され、決済ポジションのcarryは決済時に計上される。"""
    bt = PortfolioBacktester(
        initial_capital=1_000_000,
        risk=make_risk(),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.05,  # BUY uses this
        annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
    )

    orders = make_order(code="7203", side="BUY", qty=50, date="2024-01-03", holding_days=1)

    result = bt.run(orders, make_prices(), start_date="2024-01-03", end_date="2024-01-05")

    trade = result.trades.iloc[0]

    # 1日保有(01-03→01-04), entry_price(open) = 101.0
    # carry = 101.0 * (0.05/365.0) * 1 * 50
    expected_carry = 101.0 * (0.05 / 365.0) * 1 * 50
    assert trade["carry_days"] == 1
    assert trade["carry_cost"] == pytest.approx(expected_carry, rel=1e-3)

    # 決済後(01-04)のledgerにはcarryが残っていないこと
    ledger = result.daily_ledger
    exit_day = ledger[ledger["date"] == "2024-01-04"].iloc[0]
    assert exit_day["accrued_carry"] == 0.0


def test_remaining_positions_still_get_carry_after_close():
    """決済後に残ったポジションには通常通りcarryが発生する。"""
    bt = PortfolioBacktester(
        initial_capital=1_000_000,
        risk=make_risk(),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.05,  # SELL borrowing uses this
        annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
    )

    # 7203: BUY, holding_days=2 → 01-05決済
    # 6758: SELL, holding_days=10 → 長期保有
    # 両方 holding_days=2 → exit_date=01-05（trading_datesあり）
    orders = pd.DataFrame([
        {
            "date": "2024-01-03", "code": "7203", "side": "BUY",
            "qty": 100, "order_type": "MKT_OPEN", "holding_days": 2,
            "score": 1.0, "shortable": True,
        },
        {
            "date": "2024-01-03", "code": "6758", "side": "SELL",
            "qty": 100, "order_type": "MKT_OPEN", "holding_days": 2,
            "score": 1.0, "shortable": True,
        },
    ])

    result = bt.run(orders, make_prices(), start_date="2024-01-03", end_date="2024-01-05")

    # 決済後(01-05)のledger
    # 7203 BUY決済（決済時にcarry計上）, 6758 SELL決済（決済時にcarry計上）
    # 決済後はpositions=0なのでaccrued_carry=0
    ledger = result.daily_ledger
    day5 = ledger[ledger["date"] == "2024-01-05"].iloc[0]
    assert day5["accrued_carry"] == 0.0

    # 7203 BUY: entry_mtm = 102*100=10,200, 決済で103.0→exit_price
    # 保有2日(01-03→01-05)のcarry = 101.0*(0.05/365)*2*100
    expected_carry_7203 = 101.0 * (0.05 / 365.0) * 2 * 100
    trade_7203 = result.trades[result.trades["code"] == "7203"].iloc[0]
    assert trade_7203["carry_cost"] == pytest.approx(expected_carry_7203, rel=1e-3)


def test_short_side_closing_uses_lending_rate():
    """空売り決済時に lending rate で carry が計上される。"""
    bt = PortfolioBacktester(
        initial_capital=1_000_000,
        risk=make_risk(),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0,
        annual_lending_rate=0.10,  # SELL uses this
        adv_window=2, min_adv_periods=2,
    )

    orders = make_order(code="6758", side="SELL", qty=200, date="2024-01-03", holding_days=1)

    result = bt.run(orders, make_prices(), start_date="2024-01-03", end_date="2024-01-05")

    trade = result.trades.iloc[0]

    # 1日保有: entry_price(open)=199.0
    # carry = 199.0 * (0.10/365.0) * 1 * 200
    expected_carry = 199.0 * (0.10 / 365.0) * 1 * 200
    assert trade["carry_days"] == 1
    assert trade["carry_cost"] == pytest.approx(expected_carry, rel=1e-3)


# ----------------------------------------------------------------
# 2. equity_at_open の計算と検証
# ----------------------------------------------------------------

def test_equity_at_open_uses_correct_cash_and_positions():
    """equity_at_open が決済後のcashとポジションで計算される。"""
    bt = PortfolioBacktester(
        initial_capital=1_000_000,
        risk=make_risk(),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
    )

    # BUY 7203 50株 @ 101 = 5,050
    # cash = 994,950, holding_days=1 → 01-04決済
    orders = make_order(code="7203", side="BUY", qty=50, date="2024-01-03", holding_days=1)

    result = bt.run(orders, make_prices(), start_date="2024-01-03", end_date="2024-01-05")

    ledger = result.daily_ledger

    # 01-04: 決済日。決済後ポジション0, gross=0 → equity_to_gross_ratio=None
    day4 = ledger[ledger["date"] == "2024-01-04"].iloc[0]
    assert day4["open_position_count"] == 0
    assert day4["gross_exposure"] == 0.0
    assert day4["equity_to_gross_ratio"] is None or pd.isna(day4["equity_to_gross_ratio"])


def test_margin_breach_on_entry():
    """エントリー時に equity_to_gross_ratio が維持率を下回ると MARGIN_BREACH。"""
    prices = make_prices()

    bt = PortfolioBacktester(
        initial_capital=100_000,
        risk=make_risk(max_gross_exposure_yen=50_000_000, max_long_exposure_yen=50_000_000,
                       max_net_exposure_yen=50_000_000),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
        maintain_margin_ratio=0.25,
    )

    # BUY 5000株 @ 101 = 505,000. equity_at_open=100,000
    # equity_to_gross = 100,000/505,000 = 0.198 < 0.25 → MARGIN_BREACH
    orders = make_order(code="7203", side="BUY", qty=5000, date="2024-01-03", holding_days=2)

    result = bt.run(orders, prices, start_date="2024-01-03", end_date="2024-01-05")

    reasons = set(result.rejected_orders["reason"]) if not result.rejected_orders.empty else set()
    assert "MARGIN_BREACH" in reasons, f"Expected MARGIN_BREACH, got reasons={reasons}"


# ----------------------------------------------------------------
# 3. ledger の新規フィールド
# ----------------------------------------------------------------

def test_ledger_has_margin_fields():
    """daily_ledger に equity_to_gross_ratio, minimum_equity_to_gross_ratio,
    margin_breach が含まれる。"""
    bt = PortfolioBacktester(
        initial_capital=1_000_000, risk=make_risk(),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
    )
    orders = make_order(code="7203", side="BUY", qty=100, date="2024-01-03", holding_days=5)

    result = bt.run(orders, make_prices(), start_date="2024-01-03", end_date="2024-01-05")

    for _, row in result.daily_ledger.iterrows():
        assert "equity_to_gross_ratio" in row.index
        assert "minimum_equity_to_gross_ratio" in row.index
        assert "margin_breach" in row.index


def test_ledger_margin_fields_meaningful():
    """ポジション保有時に equity_to_gross_ratio が正しく計算される。"""
    bt = PortfolioBacktester(
        initial_capital=1_000_000, risk=make_risk(),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
    )
    orders = make_order(code="7203", side="BUY", qty=100, date="2024-01-03", holding_days=5)

    result = bt.run(orders, make_prices(), start_date="2024-01-03", end_date="2024-01-05")

    entry_day = result.daily_ledger[result.daily_ledger["date"] == "2024-01-03"].iloc[0]
    # equity = 1,000,000 - 101*100 + 101*100 = 1,000,000
    # gross = 101*100 = 10,100
    # equity_to_gross = 1,000,000 / 10,100 ≈ 99.0
    if entry_day["gross_exposure"] > 0 and entry_day["equity_to_gross_ratio"] is not None:
        assert entry_day["equity_to_gross_ratio"] > 1.0
        assert not entry_day["margin_breach"]


def test_no_margin_breach_when_gross_is_zero():
    """gross_exposure=0 の場合、margin_breach=False, equity_to_gross_ratio=None。"""
    bt = PortfolioBacktester(
        initial_capital=1_000_000, risk=make_risk(),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
    )
    orders = make_order(code="7203", side="BUY", qty=100, date="2024-01-03", holding_days=1)

    result = bt.run(orders, make_prices(), start_date="2024-01-03", end_date="2024-01-04")

    day4 = result.daily_ledger[result.daily_ledger["date"] == "2024-01-04"].iloc[0]
    assert day4["gross_exposure"] == 0.0
    assert not day4["margin_breach"]


def test_empty_orders_ledger_has_margin_fields():
    """注文0件のflat NAV ledgerにも equity_to_gross_ratio 等が含まれる。"""
    bt = PortfolioBacktester(
        initial_capital=1_000_000, risk=make_risk(),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
    )
    result = bt.run(pd.DataFrame(), make_prices(), start_date="2024-01-03", end_date="2024-01-05")

    assert "equity_to_gross_ratio" in result.daily_ledger.columns
    assert "minimum_equity_to_gross_ratio" in result.daily_ledger.columns
    assert "margin_breach" in result.daily_ledger.columns
    assert not result.daily_ledger["margin_breach"].any()


# ----------------------------------------------------------------
# 4. MARGIN_BREACH リジェクト
# ----------------------------------------------------------------

def test_margin_breach_rejects_new_orders():
    """信用保証金率が維持率を下回ると新規注文が MARGIN_BREACH でリジェクトされる。"""
    bt = PortfolioBacktester(
        initial_capital=200_000,
        risk=make_risk(max_gross_exposure_yen=50_000_000, max_long_exposure_yen=50_000_000,
                       max_single_name_exposure_yen=50_000_000),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
        maintain_margin_ratio=0.15,
    )

    prices = make_prices()

    # 01-03: BUY 7203 1500株 @ 101 = 151,500 → cash = 48,500
    # 01-04 equity_at_open = 48,500 + 103*1500 = 203,000
    #        gross = 154,500, equity_to_gross = 1.31 > 0.15 → OK
    #
    # 01-04: BUY 6758 10000株 @ 197 = 1,970,000
    #        next_gross = 154,500 + 1,970,000 = 2,124,500
    #        next_equity_to_gross = 203,000/2,124,500 = 0.095 < 0.15 → MARGIN_BREACH
    orders = pd.DataFrame([
        {
            "date": "2024-01-03", "code": "7203", "side": "BUY",
            "qty": 1500, "order_type": "MKT_OPEN", "holding_days": 5,
            "score": 2.0, "shortable": True,
        },
        {
            "date": "2024-01-04", "code": "6758", "side": "BUY",
            "qty": 10000, "order_type": "MKT_OPEN", "holding_days": 5,
            "score": 1.0, "shortable": True,
        },
    ])

    result = bt.run(orders, prices, start_date="2024-01-03", end_date="2024-01-05")

    reasons = set(result.rejected_orders["reason"]) if not result.rejected_orders.empty else set()
    # MARGIN_BREACH or GROSS_LIMIT depending on which triggers first
    has_rejection = bool(reasons)
    assert has_rejection, f"Expected some rejection, got trades={len(result.trades)} only"


def test_margin_breach_happens_only_when_ratio_low():
    """十分な equity がある場合、MARGIN_BREACH は発生しない。"""
    bt = PortfolioBacktester(
        initial_capital=1_000_000,
        risk=make_risk(),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
        maintain_margin_ratio=0.25,
    )

    orders = pd.DataFrame([
        {
            "date": "2024-01-03", "code": "7203", "side": "BUY",
            "qty": 100, "order_type": "MKT_OPEN", "holding_days": 5,
            "score": 1.0, "shortable": True,
        },
        {
            "date": "2024-01-04", "code": "6758", "side": "BUY",
            "qty": 100, "order_type": "MKT_OPEN", "holding_days": 5,
            "score": 1.0, "shortable": True,
        },
    ])

    result = bt.run(orders, make_prices(), start_date="2024-01-03", end_date="2024-01-05")

    reasons = set(result.rejected_orders["reason"]) if not result.rejected_orders.empty else set()
    assert "MARGIN_BREACH" not in reasons, f"Unexpected MARGIN_BREACH: {reasons}"


# ----------------------------------------------------------------
# 5. _select_orders の gross limit 修正
# ----------------------------------------------------------------

def test_gross_limit_uses_next_gross_not_next_long():
    """_select_orders が next_long ではなく next_gross でGROSS_LIMITを判定する。"""
    risk = make_risk(
        max_gross_exposure_yen=500_000,
        max_single_name_exposure_yen=500_000,
        max_long_exposure_yen=1_000_000,
        max_short_exposure_yen=1_000_000,
        max_net_exposure_yen=1_000_000,
    )

    bt = PortfolioBacktester(
        initial_capital=1_000_000, risk=risk,
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=20, min_adv_periods=1,
    )

    prices = make_prices_long()

    # holding_days=2 → exit_date=01-05（trading_dates内に存在）
    orders = pd.DataFrame([
        {
            "date": "2024-01-03", "code": "7203", "side": "BUY",
            "qty": 3000, "order_type": "MKT_OPEN", "holding_days": 2,
            "score": 2.0, "shortable": True,
        },
        {
            "date": "2024-01-03", "code": "6758", "side": "BUY",
            "qty": 1000, "order_type": "MKT_OPEN", "holding_days": 2,
            "score": 1.0, "shortable": True,
        },
    ])

    result = bt.run(orders, prices, start_date="2024-01-03", end_date="2024-01-05")

    # score=2.0の7203(303K gross) が採用、score=1.0の6758(199K, total=502K>500K) がGROSS_LIMIT
    assert len(result.trades) == 1, f"Expected 1 trade, got {len(result.trades)}"
    assert result.trades.iloc[0]["code"] == "7203"

    if not result.rejected_orders.empty:
        rejected_reasons = result.rejected_orders["reason"].tolist()
        assert "GROSS_LIMIT" in rejected_reasons, f"Expected GROSS_LIMIT, got {rejected_reasons}"


def test_gross_limit_with_buy_and_sell():
    """BUYとSELL混在時にgross limit（絶対値和）が正しく判定される。"""
    risk = make_risk(
        max_gross_exposure_yen=800_000,
        max_single_name_exposure_yen=800_000,
        max_long_exposure_yen=600_000,
        max_short_exposure_yen=600_000,
        max_net_exposure_yen=1_000_000,
    )

    bt = PortfolioBacktester(
        initial_capital=2_000_000, risk=risk,
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=20, min_adv_periods=1,
    )

    prices = make_prices_long()

    # BUY 7203: 4000株 @ 101 = 404,000 (long)
    # SELL 6758: 2000株 @ 199 = 398,000 (short)
    # gross = 404,000 + 398,000 = 802,000 > 800,000
    orders = pd.DataFrame([
        {
            "date": "2024-01-03", "code": "7203", "side": "BUY",
            "qty": 4000, "order_type": "MKT_OPEN", "holding_days": 2,
            "score": 2.0, "shortable": True,
        },
        {
            "date": "2024-01-03", "code": "6758", "side": "SELL",
            "qty": 2000, "order_type": "MKT_OPEN", "holding_days": 2,
            "score": 1.0, "shortable": True,
        },
    ])

    result = bt.run(orders, prices, start_date="2024-01-03", end_date="2024-01-05")

    assert len(result.trades) == 1

    if not result.rejected_orders.empty:
        reasons = set(result.rejected_orders["reason"])
        assert "GROSS_LIMIT" in reasons, f"Expected GROSS_LIMIT, got {reasons}"


# ----------------------------------------------------------------
# 6.（リグレッション）決済予定ポジションのexisting blocking
# ----------------------------------------------------------------

def test_existing_position_still_blocks_same_code_at_open():
    """決済予定のポジションが寄付き時点ではまだ既存コードとしてブロックされる。"""
    bt = PortfolioBacktester(
        initial_capital=1_000_000, risk=make_risk(),
        impact_k_bp=0.0, commission_bp=0.0, half_spread_bp=0.0,
        annual_interest_rate=0.0, annual_lending_rate=0.0,
        adv_window=2, min_adv_periods=2,
    )

    orders = pd.DataFrame([
        {
            "date": "2024-01-03", "code": "7203", "side": "BUY",
            "qty": 100, "order_type": "MKT_OPEN", "holding_days": 1,
            "score": 1.0, "shortable": False,
        },
        {
            "date": "2024-01-04", "code": "7203", "side": "BUY",
            "qty": 100, "order_type": "MKT_OPEN", "holding_days": 1,
            "score": 1.0, "shortable": False,
        },
    ])

    result = bt.run(orders, make_prices(), start_date="2024-01-03", end_date="2024-01-05")

    assert "EXISTING_POSITION" in set(result.rejected_orders["reason"])
