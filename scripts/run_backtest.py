"""バックテスト実行スクリプト。

使い方: python scripts/run_backtest.py

P0:
  - yfinance 近似 turnover 利用時は hard fail（明示オプトイン以外）
  - shortability 未確認売りはデフォルト除外
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jp_signal.backtest import Backtester
from jp_signal.config import guard_approximate_turnover, load_config
from jp_signal.metrics import summarize_backtest
from jp_signal.model import MeanReversionRule
from jp_signal.order_builder import signals_to_orders
from jp_signal.risk import risk_config_from_dict
from jp_signal.storage import Storage
from jp_signal.universe import load_universe


def main() -> None:
    cfg = load_config()

    # P0: 近似 turnover で impact/sizing を使う BT を拒否
    guard_approximate_turnover(cfg, context="run_backtest")

    if not cfg.get("backtest", {}).get("impact_k_is_calibrated", False):
        print(
            "[WARN] impact_k_is_calibrated=false: "
            "インパクト係数は未較正です。PnL を過信しないでください。"
        )

    with Storage(cfg["data"]["db_path"]) as st:
        univ_all = load_universe(cfg["universe"])
        codes = univ_all["code"].tolist()

        bt_start = pd.Timestamp(cfg["backtest"]["start"]).date()

        bt_end = pd.Timestamp(cfg["backtest"]["end"]).date()

        model_lookback = int(cfg.get("model", {}).get("lookback", 5))

        adv_window = int(cfg["backtest"].get("adv_window", 20))

        # 営業日数より十分長い暦日バッファ
        warmup_calendar_days = (
            max(
                model_lookback,
                adv_window,
            )
            * 3
        )

        load_start = bt_start - timedelta(days=warmup_calendar_days)

        prices = st.load_prices(
            codes,
            load_start.isoformat(),
            bt_end.isoformat(),
        )

        short = st.load_shortability(
            codes,
            load_start.isoformat(),
            bt_end.isoformat(),
        )

        if prices.empty:
            print("価格データが空です。先に main.py 等で DB へ取り込んでください。")
            return

        if short.empty:
            print(
                "[WARN] shortability データが空です。"
                " 売りは SKIP_NOT_SHORTABLE になります"
                "（allow_unconfirmed_short_in_bt / "
                "allow_short_without_confirmed_shortability が true の場合を除く）。"
            )

        model = MeanReversionRule(
            lookback=int(cfg.get("model", {}).get("lookback", 5)),
            top_n=int(cfg.get("model", {}).get("top_n", 5)),
        )
        holding_days = int(cfg["backtest"].get("holding_days", 1))

        # BT 用 short ポリシー:
        # - デフォルトは live と同じく未確認売り禁止
        # - 開発時のみ backtest.allow_unconfirmed_short_in_bt=true
        risk_cfg = risk_config_from_dict(cfg.get("risk", {}))
        if cfg.get("backtest", {}).get("allow_unconfirmed_short_in_bt", False):
            risk_cfg.allow_short_without_confirmed_shortability = True
            print(
                "[WARN] allow_unconfirmed_short_in_bt=true: "
                "BT で shortability 未確認売りを許可します（開発専用）。"
            )
        else:
            risk_cfg.allow_short_without_confirmed_shortability = False

        all_dates = sorted(
            d
            for d in prices["date"].unique()
            if cfg["backtest"]["start"] <= d <= cfg["backtest"]["end"]
        )

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

        summary = summarize_backtest(
            result,
            initial_capital=float(cfg["backtest"].get("initial_capital", 100_000_000)),
            risk_free_rate=float(cfg["backtest"].get("risk_free_rate", 0.0)),
            trading_dates=all_dates,
        )
        print(f"全シグナル: {summary['n_signals']}")
        print(f"約定: {summary['n_filled']}")
        print(f"約定率: {summary['fill_rate'] * 100:.1f}%")
        print(f"合計PnL: {summary['total_pnl']:,.0f}円")
        print(f"総リターン: {summary['total_return'] * 100:.2f}%")
        print(f"勝率: {summary['win_rate'] * 100:.1f}%")
        print(f"Sharpe: {summary['sharpe']:.2f}")
        print(
            "最大DD: "
            f"{summary['max_drawdown_yen']:,.0f}円 "
            f"({summary['max_drawdown_pct'] * 100:.2f}%)"
        )
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
