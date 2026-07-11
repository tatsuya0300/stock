"""コーポレートアクション処理（SPLIT / CASH_DIVIDEND）。

制約:
  - SPLIT: qty と entry_price を調整する（エントリー時点の価格基準を維持）
  - CASH_DIVIDEND: 長期保有は受取、空売りは支払い
  - 複合アクション（SPLIT + DIVIDEND 同時など）は非対応
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

ActionType = Literal["SPLIT", "CASH_DIVIDEND"]


@dataclass(frozen=True)
class CorporateAction:
    """コーポレートアクション1件。

    Attributes:
        code: 証券コード
        ex_date: 権利落ち日
        action_type: SPLIT または CASH_DIVIDEND
        ratio: SPLIT の場合の分割比率（例: 5→1分割なら 5.0）
        amount: CASH_DIVIDEND の場合の一株あたり配当額（円）
    """

    code: str
    ex_date: str
    action_type: ActionType
    ratio: float | None = None
    amount: float | None = None


def prepare_corporate_actions(df: pd.DataFrame) -> list[CorporateAction]:
    """DataFrame からコーポレートアクションリストを生成する。

    CSVなどの外部データから読み込んだDataFrameを想定。
    必須列: code, ex_date, action_type
    任意列: ratio（SPLIT時）, amount（CASH_DIVIDEND時）

    Args:
        df: コーポレートアクション一覧のDataFrame

    Returns:
        CorporateAction のリスト（ex_date 昇順）
    """
    if df is None or df.empty:
        return []

    required = {"code", "ex_date", "action_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"corporate_actions missing columns: {sorted(missing)}")

    actions: list[CorporateAction] = []
    for _, row in df.iterrows():
        action_type = str(row["action_type"]).upper()
        if action_type not in ("SPLIT", "CASH_DIVIDEND"):
            continue

        actions.append(
            CorporateAction(
                code=str(row["code"]).strip(),
                ex_date=str(pd.to_datetime(row["ex_date"]).date()),
                action_type=action_type,
                ratio=float(row["ratio"]) if pd.notna(row.get("ratio")) else None,
                amount=float(row["amount"]) if pd.notna(row.get("amount")) else None,
            )
        )

    return sorted(actions, key=lambda a: a.ex_date)


def _apply_corporate_actions(
    positions: list,
    actions: list[CorporateAction],
    ex_date: str,
    cash: float,
    ledger: list[dict] | None = None,
) -> tuple[list, float]:
    """コーポレートアクションをポジションに適用する。

    SPLIT:
      - qty を ratio 倍、entry_price を 1/ratio 倍する
      - エントリー時点の価値基準を維持する

    CASH_DIVIDEND:
      - 長期保有（BUY）は amount * qty を受取（cash増加）
      - 空売り（SELL）は amount * qty を支払い（cash減少）
      - ledger に dividend_cashflow として記録

    Args:
        positions: ポジションリスト（Position オブジェクト）
        actions: 該当日のコーポレートアクションリスト
        ex_date: 権利落ち日（YYYY-MM-DD）
        cash: 現在の現金残高
        ledger: 日次台帳リスト（オプション、dividend_cashflow記録用）

    Returns:
        (更新されたポジションリスト, 更新されたcash)
    """
    if not actions:
        return positions, cash

    updated_positions = []

    # 該当銘柄のアクションをグループ化
    action_map: dict[str, list[CorporateAction]] = {}
    for a in actions:
        action_map.setdefault(a.code, []).append(a)

    for pos in positions:
        code = str(pos.code)
        if code not in action_map:
            updated_positions.append(pos)
            continue

        # Position を可変 dict に変換して処理
        pos_dict = {
            "qty": pos.qty,
            "entry_price": pos.entry_price,
            "accrued_carry": pos.accrued_carry if hasattr(pos, "accrued_carry") else 0.0,
        }

        total_dividend = 0.0

        for action in action_map[code]:
            if action.action_type == "SPLIT" and action.ratio and action.ratio > 0:
                pos_dict["qty"] = int(pos_dict["qty"] * action.ratio)
                pos_dict["entry_price"] = pos_dict["entry_price"] / action.ratio

            elif action.action_type == "CASH_DIVIDEND" and action.amount:
                if pos.side == "BUY":
                    total_dividend += action.amount * pos_dict["qty"]
                elif pos.side == "SELL":
                    total_dividend -= action.amount * pos_dict["qty"]

        # 配当金のcash反映
        if total_dividend != 0:
            cash += total_dividend
            if ledger is not None:
                ledger.append(
                    {
                        "date": ex_date,
                        "code": code,
                        "dividend_cashflow": total_dividend,
                        "type": "DIVIDEND",
                    }
                )

        # qty/entry_price が変わった場合、新しい Position を作成
        if pos_dict["qty"] != pos.qty or pos_dict["entry_price"] != pos.entry_price:
            from dataclasses import replace

            updated_positions.append(
                replace(
                    pos,
                    qty=pos_dict["qty"],
                    entry_price=pos_dict["entry_price"],
                )
            )
        else:
            updated_positions.append(pos)

    return updated_positions, cash
