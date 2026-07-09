"""バックテスト実行スクリプト。

使い方: python scripts/run_backtest.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jp_signal.backtest import Backtester
from jp_signal.config import load_config
from jp_signal.metrics import summarize_backtest
from jp_signal.model import MeanReversionRule
from jp_signal.order_builder import signals_to_orders
from jp_signal.risk import risk_config_from_dict
from jp_signal.storage import Storage
from jp_signal.universe import load_universe


def main() -> None:
    cfg = load_config()
    with Storage(cfg["data"]["db_path"]) as st:
        univ_all = load_universe(cfg["universe"])
        codes = univ_all["code"].tolist()

        prices = st.load_prices(
            codes, cfg["backtest"]["start"], cfg["backtest"]["end"]
        )
        short = st.load_shortability(
            codes, cfg["backtest"]["start"], cfg["backtest"]["end"]
        )
        if prices.empty:
            print("価格データが空です。先に main.py 等で DB へ取り込んでください。")
            return

        model = MeanReversionRule(
            lookback=int(cfg.get("model", {}).get("lookback", 5)),
            top_n=int(cfg.get("model", {}).get("top_n", 5)),
        )
        holding_days = int(cfg["backtest"].get("holding_days", 1))
        risk_cfg = risk_config_from_dict(cfg.get("risk", {}))
        all_dates = sorted(prices["date"].unique())

        signal_frames: list[pd.DataFrame] = []
        for d in all_dates[20:]:
            univ_d = load_universe(cfg["universe"], as_of=d)
            codes_d = set(univ_d["code"].tolist())
            px_d = prices[prices["code"].isin(codes_d)]
            if px_d.empty:
                continue

            sig = model.generate(px_d, as_of=d)
            if sig.empty:
                continue

            day_orders = signals_to_orders(
                sig,
                px_d,
                as_of=d,
                sizing_cfg=cfg["sizing"],
                risk_cfg=risk_cfg,
                shortability=short if not short.empty else None,
                universe=univ_d,
                holding_days=holding_days,
                order_type="MKT_OPEN",
                unit=int(cfg.get("sizing", {}).get("unit", 100)),
                for_backtest=True,
            )
            if not day_orders.empty:
                signal_frames.append(day_orders)

        if not signal_frames:
            print("シグナル0件。バックテストをスキップします。")
            return

        signals = pd.concat(signal_frames, ignore_index=True)
        # Backtester 必須列の保証
        if "limit_price" not in signals.columns:
            signals["limit_price"] = None

        bt = Backtester(
            impact_k_bp=float(cfg["backtest"].get("impact_k_bp", 30.0)),
            annual_interest_rate=float(cfg["backtest"].get("annual_interest_rate", 0.02)),
            annual_lending_rate=float(cfg["backtest"].get("short_lending_rate", 0.02)),
            commission_bp=float(cfg["backtest"].get("commission_bp", 0.0)),
            half_spread_bp=float(cfg["backtest"].get("half_spread_bp", 0.0)),
            adv_window=int(cfg["backtest"].get("adv_window", 20)),
            require_liquidity_data=True,
            zero_carry_for_intraday=True,
        )
        result = bt.run(
            signals,
            prices,
            shortability=short if not short.empty else None,
        )
        if result.empty:
            print("バックテスト結果が空です。")
            return

        summary = summarize_backtest(result)
        print(f"全シグナル: {summary['n_signals']}")
        print(f"約定: {summary['n_filled']}")
        print(f"約定率: {summary['fill_rate'] * 100:.1f}%")
        print(f"合計PnL: {summary['total_pnl']:.0f}")
        print(f"勝率: {summary['win_rate'] * 100:.1f}%")
        print(f"日次PnL平均  : {summary['daily_pnl_mean']:.0f}")
        print(f"日次PnL標準偏差: {summary['daily_pnl_std']:.0f}")
        print(f"Sharpe-like: {summary['sharpe_like']:.2f}")
        print(f"MaxDD(PnL): {summary['max_drawdown_pnl']:.0f}")
        print(f"status: {summary['status_counts']}")

        # 成果物
        out_dir = cfg.get("backtest", {}).get("output_dir", "./data/bt_out")
        os.makedirs(out_dir, exist_ok=True)
        result.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
        if "daily_pnl" in summary and summary["daily_pnl"] is not None:
            summary["daily_pnl"].to_csv(os.path.join(out_dir, "daily_pnl.csv"), header=["pnl"])
        print(f"wrote: {out_dir}/trades.csv")


if __name__ == "__main__":
    main()
