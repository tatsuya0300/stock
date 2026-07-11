"""corporate_actions のテスト。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from jp_signal.corporate_actions import (
    CorporateAction,
    _apply_corporate_actions,
    prepare_corporate_actions,
)


@dataclass
class FakePosition:
    code: str
    side: str
    qty: int
    entry_price: float
    accrued_carry: float = 0.0
    name: str = ""


def test_prepare_corporate_actions_empty():
    assert prepare_corporate_actions(pd.DataFrame()) == []


def test_prepare_corporate_actions_sorts_by_date():
    df = pd.DataFrame(
        [
            {"code": "7203", "ex_date": "2024-02-01", "action_type": "SPLIT", "ratio": 5.0},
            {
                "code": "7203",
                "ex_date": "2024-01-15",
                "action_type": "CASH_DIVIDEND",
                "amount": 50.0,
            },
        ]
    )
    actions = prepare_corporate_actions(df)
    assert len(actions) == 2
    assert actions[0].action_type == "CASH_DIVIDEND"  # 1月が先


def test_apply_split():
    pos = FakePosition(code="7203", side="BUY", qty=100, entry_price=5000.0)
    action = CorporateAction(code="7203", ex_date="2024-06-01", action_type="SPLIT", ratio=5.0)
    updated, cash, events = _apply_corporate_actions([pos], [action], "2024-06-01", 1_000_000.0)
    assert len(updated) == 1
    assert updated[0].qty == 500
    assert updated[0].entry_price == 1000.0
    assert len(events) == 0  # SPLIT does not create events


def test_dividend_long_receives_cash():
    pos = FakePosition(code="7203", side="BUY", qty=100, entry_price=5000.0)
    action = CorporateAction(
        code="7203", ex_date="2024-03-01", action_type="CASH_DIVIDEND", amount=50.0
    )
    updated, cash, events = _apply_corporate_actions([pos], [action], "2024-03-01", 1_000_000.0)
    assert cash == 1_005_000.0  # +50*100
    assert len(updated) == 1
    assert len(events) == 1
    assert events[0]["action_type"] == "CASH_DIVIDEND"


def test_dividend_short_pays_cash():
    pos = FakePosition(code="7203", side="SELL", qty=100, entry_price=5000.0)
    action = CorporateAction(
        code="7203", ex_date="2024-03-01", action_type="CASH_DIVIDEND", amount=50.0
    )
    updated, cash, events = _apply_corporate_actions([pos], [action], "2024-03-01", 1_000_000.0)
    assert cash == 995_000.0  # -50*100
    assert len(events) == 1
    assert events[0]["cashflow"] == -5000.0


def test_unrelated_code_unchanged():
    pos = FakePosition(code="6758", side="BUY", qty=100, entry_price=5000.0)
    action = CorporateAction(
        code="7203", ex_date="2024-03-01", action_type="CASH_DIVIDEND", amount=50.0
    )
    updated, cash, events = _apply_corporate_actions([pos], [action], "2024-03-01", 1_000_000.0)
    assert cash == 1_000_000.0
    assert len(updated) == 1
    assert updated[0].qty == 100
    assert len(events) == 0  # unrelated code
