"""バックテスト実行スクリプト。

DB に蓄積済みの価格を用いて MeanReversionRule のシグナルをバックテストする。
使い方: python scripts/run_backtest.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# プロジェクトルートを import パスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jp_signal.backtest import Backtester
from jp_signal.model import MeanReversionRule
from jp_signal.pipeline import load_config
from jp_signal.storage import Storage
from jp_signal.universe import load_universe


def main() -> None:
    cfg = load_config()
    st = Storage(cfg["data"]["db_path"])
    univ = load_universe(cfg["universe"]["file"])
    codes = univ["code"].tolist()

    prices = st.load_prices(codes, cfg["backtest"]["start"], cfg["backtest"]["end"])
    short = st.load_shortability(codes, cfg["backtest"]["start"], cfg["backtest"]["end"])
    if prices.empty:
        print("価格データが空です。先に main.py 等で DB へ取り込んでください。")
        return

    model = MeanReversionRule(lookback=5, top_n=5)
    all_dates = sorted(prices["date"].unique())

    signals = []
    for d in all_dates[20:]:  # 十分な履歴が溜まってから
        sig = model.generate(prices, as_of=d)
        if sig.empty:
            continue
        sig = sig.copy()
        sig["date"] = d
        sig["order_type"] = "MKT_OPEN"
        sig["qty"] = 100          # 固定単元で検証
        sig["holding_days"] = 1
        signals.append(sig)

    if not signals:
        print("シグナルが生成されませんでした。")
        return
    signals = pd.concat(signals, ignore_index=True)

    bt = Backtester(
        impact_k_bp=cfg["backtest"]["impact_k_bp"],
        annual_interest_rate=cfg["backtest"]["annual_interest_rate"],
        annual_lending_rate=cfg["backtest"]["short_lending_rate"],
    )
    result = bt.run(signals, prices, shortability=short if not short.empty else None)

    print("=== ステータス別件数 ===")
    print(result.groupby("status").size())
    filled = result.query("status=='FILLED'")
    if not filled.empty:
        print("\n=== サマリ ===")
        print(f"約定件数     : {len(filled)}")
        print(f"合計PnL      : {filled['pnl'].sum():.0f}")
        print(f"平均PnL/取引 : {filled['pnl'].mean():.2f}")
        print(f"勝率         : {(filled['pnl'] > 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
