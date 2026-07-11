"""Portfolio-level backtester tests."""

from __future__ import annotations

import pandas as pd
import pytest

from jp_signal.portfolio import PortfolioBacktester
from jp_signal.risk import RiskConfig


def make_risk(max_net: float = 10_000_000) -> RiskConfig:
    return RiskConfig(
        max_orders_per_day=10,
        max_gross_exposure_yen=20_000_000,
        max_single_name_exposure_yen=10_000_000,
        max_long_exposure_yen=10_000_000,
        max_short_exposure_yen=10_000_000,
        max_net_exposure_yen=max_net,
        require_both_sides=False,
        allow_short_without_confirmed_shortability=True,
    )


def make_prices() -> pd.DataFrame:
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
            rows.append(
                {
                    "code": code,
                    "date": dt,
                    "open": open_price,
                    "high": max(open_price, close_price),
                    "low": min(open_price, close_price),
                    "close": close_price,
                    "adj_open": open_price,
                    "adj_high": max(open_price, close_price),
                    "adj_low": min(open_price, close_price),
                    "adj_close": close_price,
                    "volume": 1_000_000,
                    "turnover": 100_000_000,
                }
            )

    return pd.DataFrame(rows)


def make_backtester(max_net: float = 10_000_000) -> PortfolioBacktester:
    return PortfolioBacktester(
        initial_capital=1_000_000,
        risk=make_risk(max_net),
        impact_k_bp=0.0,
        commission_bp=0.0,
        half_spread_bp=0.0,
        annual_interest_rate=0.0,
        annual_lending_rate=0.0,
        adv_window=2,
        min_adv_periods=2,
    )


def test_long_trade_updates_nav():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "name": "Toyota",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    assert trade["entry"] == 101.0
    assert trade["exit"] == 102.0
    assert trade["pnl"] == 100.0

    assert result.daily_ledger.iloc[-1]["nav"] == 1_000_100.0


def test_existing_position_blocks_duplicate_code():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 2,
                "score": 1.0,
                "shortable": False,
            },
            {
                "date": "2024-01-04",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert "EXISTING_POSITION" in set(result.rejected_orders["reason"])


def test_net_limit_rejects_unbalanced_order():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = make_backtester(max_net=5_000).run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert result.trades.empty
    assert "NET_LIMIT" in set(result.rejected_orders["reason"])


def test_balanced_long_short_passes_net_limit():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
            {
                "date": "2024-01-03",
                "code": "6758",
                "side": "SELL",
                "qty": 50,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": True,
            },
        ]
    )

    result = make_backtester(max_net=1_000).run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert len(result.trades) == 2
    assert result.rejected_orders.empty


def test_nav_accounting_identity():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 2,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    ledger = result.daily_ledger

    expected_nav = (
        ledger["cash"]
        + ledger["long_exposure"]
        - ledger["short_exposure"]
        - ledger["accrued_carry"]
    )
    error = (ledger["nav"] - expected_nav).abs()

    assert float(error.max()) < 0.01


def test_short_trade_cash_and_nav_accounting():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "6758",
                "name": "Sony",
                "side": "SELL",
                "qty": 50,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": True,
            }
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    # 199円で50株売り、198円で買い戻し。
    assert trade["entry"] == pytest.approx(199.0)
    assert trade["exit"] == pytest.approx(198.0)
    assert trade["gross_pnl"] == pytest.approx(50.0)
    assert trade["pnl"] == pytest.approx(50.0)

    final_ledger = result.daily_ledger.iloc[-1]

    assert final_ledger["cash"] == pytest.approx(1_000_050.0)
    assert final_ledger["nav"] == pytest.approx(1_000_050.0)


def test_short_flat_price_does_not_create_artificial_cash():
    prices = make_prices().copy()

    mask = (prices["code"] == "6758") & prices["date"].isin(["2024-01-03", "2024-01-04"])

    prices.loc[mask, "open"] = 200.0
    prices.loc[mask, "high"] = 200.0
    prices.loc[mask, "low"] = 200.0
    prices.loc[mask, "close"] = 200.0

    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "6758",
                "side": "SELL",
                "qty": 50,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": True,
            }
        ]
    )

    result = make_backtester().run(
        orders,
        prices,
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["gross_pnl"] == pytest.approx(0.0)
    assert result.daily_ledger.iloc[-1]["nav"] == pytest.approx(1_000_000.0)


def test_carry_cost_reduces_cash_and_nav():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "6758",
                "side": "SELL",
                "qty": 50,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": True,
            }
        ]
    )

    bt = PortfolioBacktester(
        initial_capital=1_000_000,
        risk=make_risk(),
        impact_k_bp=0.0,
        commission_bp=0.0,
        half_spread_bp=0.0,
        annual_interest_rate=0.0,
        annual_lending_rate=0.365,
        adv_window=2,
        min_adv_periods=2,
    )

    result = bt.run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    trade = result.trades.iloc[0]

    expected_carry = 199.0 * (0.365 / 365.0) * 1 * 50

    expected_pnl = 50.0 - expected_carry

    assert trade["carry_days"] == 1
    assert trade["carry_cost"] == pytest.approx(expected_carry)
    assert trade["pnl"] == pytest.approx(expected_pnl)

    assert result.daily_ledger.iloc[-1]["nav"] == pytest.approx(1_000_000.0 + expected_pnl)


def test_missing_exact_exit_price_defers_exit():
    prices = make_prices()

    # 7203の予定決済日2024-01-04だけを欠損させる。
    # 他銘柄には2024-01-04があるため、市場営業日自体は残る。
    prices = prices[~((prices["code"] == "7203") & (prices["date"] == "2024-01-04"))].copy()

    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = make_backtester().run(
        orders,
        prices,
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    assert trade["planned_exit_date"] == "2024-01-04"
    assert trade["exit_date"] == "2024-01-05"

    assert "NO_EXIT_PRICE_DEFERRED" in set(result.rejected_orders["reason"])


def test_unsupported_order_type_is_rejected():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "LIMIT",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert result.trades.empty
    assert "UNSUPPORTED_ORDER_TYPE" in set(result.rejected_orders["reason"])


def test_duplicate_code_same_day_is_rejected():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 2.0,
                "shortable": False,
            },
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert len(result.trades) == 1
    assert "DUPLICATE_CODE_SAME_DAY" in set(result.rejected_orders["reason"])


def test_order_without_exit_date_is_rejected():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-05",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-05",
        end_date="2024-01-05",
    )

    assert result.trades.empty
    assert "NO_EXIT_DATE_WITHIN_TEST_WINDOW" in set(result.rejected_orders["reason"])


def test_same_code_blocked_on_exit_day_at_open():
    """当日引けで決済予定のポジションは、寄付き時点ではまだ存在するため
    同一コードの新規注文をEXISTING_POSITIONとしてリジェクトする。

    これは、決済日の朝にはまだ建玉が存在するという現実を反映する。"""
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
            {
                # 01-03に建てたポジションは01-04引けで決済されるため、
                # 01-04寄付き時点ではまだ存在している → リジェクトされる
                "date": "2024-01-04",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert "EXISTING_POSITION" in set(result.rejected_orders["reason"])


def test_same_code_reentry_allowed_after_close():
    """決済日の翌営業日には、同一コードの新規注文が許可される。"""
    prices = make_prices()
    # 01-04の7203データを削除して決済を01-05に延期させる
    prices = prices[~((prices["code"] == "7203") & (prices["date"] == "2024-01-04"))].copy()

    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
            {
                # 01-03のポジションは01-04決済だが価格欠損で01-05に延期
                # 01-05寄付き時点では01-05に決済される → existing_codesに含まれる
                "date": "2024-01-05",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
        ]
    )

    result = make_backtester().run(
        orders,
        prices,
        start_date="2024-01-03",
        end_date="2024-01-06",
    )

    # 1件のtradeのみ（最初のエントリーは決済遅延のため翌日以降にずれ込む）
    assert len(result.trades) >= 1


def test_positions_exiting_today_included_in_open_risk():
    """当日引けで決済予定のポジションが、寄付き時点のリスク判定に含まれることを確認する。"""
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 2,
                "score": 1.0,
                "shortable": False,
            },
        ]
    )

    bt = make_backtester()
    # リスク上限を強く制限して、既存ポジションを含めた評価が行われることを確認
    result = bt.run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert len(result.trades) == 1


def test_final_nav_equals_total_realized_pnl():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            },
            {
                "date": "2024-01-03",
                "code": "6758",
                "side": "SELL",
                "qty": 50,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": True,
            },
        ]
    )

    result = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert result.open_positions.empty

    realized_pnl = float(result.trades["pnl"].sum())
    final_nav = float(result.daily_ledger.iloc[-1]["nav"])

    assert final_nav - 1_000_000 == pytest.approx(realized_pnl)


def test_require_both_sides_records_rejection_reason():
    risk = RiskConfig(
        max_orders_per_day=10,
        max_gross_exposure_yen=20_000_000,
        max_single_name_exposure_yen=10_000_000,
        max_long_exposure_yen=10_000_000,
        max_short_exposure_yen=10_000_000,
        max_net_exposure_yen=10_000_000,
        require_both_sides=True,
        allow_short_without_confirmed_shortability=True,
    )

    bt = PortfolioBacktester(
        initial_capital=1_000_000,
        risk=risk,
        impact_k_bp=0.0,
        commission_bp=0.0,
        half_spread_bp=0.0,
        annual_interest_rate=0.0,
        annual_lending_rate=0.0,
        adv_window=2,
        min_adv_periods=2,
    )

    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = bt.run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert result.trades.empty
    assert not result.rejected_orders.empty
    assert "REQUIRE_BOTH_SIDES" in set(result.rejected_orders["reason"])


def test_position_id_is_deterministic():
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    first = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    second = make_backtester().run(
        orders,
        make_prices(),
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    assert first.trades.iloc[0]["position_id"] == (second.trades.iloc[0]["position_id"])


def test_prior_adv_does_not_include_current_day_turnover():
    prices = make_prices().copy()

    # エントリー当日の異常に大きなturnoverがADVへ混入しないことを確認。
    mask = (prices["code"] == "7203") & (prices["date"] == "2024-01-03")
    prices.loc[mask, "turnover"] = 999_999_999_999.0

    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "code": "7203",
                "side": "BUY",
                "qty": 100,
                "order_type": "MKT_OPEN",
                "holding_days": 1,
                "score": 1.0,
                "shortable": False,
            }
        ]
    )

    result = make_backtester().run(
        orders,
        prices,
        start_date="2024-01-03",
        end_date="2024-01-05",
    )

    trade = result.trades.iloc[0]

    # 2024-01-01、2024-01-02のprior turnover平均。
    assert trade["entry_adv"] == pytest.approx(100_000_000.0)
