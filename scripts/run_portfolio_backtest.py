"""ポートフォリオ単位バックテスト。

使い方:
    python scripts/run_portfolio_backtest.py
    python scripts/run_portfolio_backtest.py --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jp_signal.config import guard_approximate_turnover, load_config
from jp_signal.coverage import CoverageThresholds, validate_daily_coverage
from jp_signal.manifest import write_backtest_manifest
from jp_signal.model import MeanReversionRule
from jp_signal.order_builder import signals_to_orders
from jp_signal.portfolio import PortfolioBacktester
from jp_signal.portfolio_metrics import summarize_portfolio_ledger
from jp_signal.risk import risk_config_from_dict
from jp_signal.storage import Storage
from jp_signal.universe import load_universe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run portfolio-level backtest")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルパス",
    )
    return parser.parse_args()


def _json_safe_summary(summary: dict) -> dict:
    return {
        key: value for key, value in summary.items() if key not in {"daily_returns", "equity_curve"}
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    guard_approximate_turnover(cfg, context="run_portfolio_backtest")

    risk_cfg = risk_config_from_dict(cfg.get("risk", {}))

    allow_unconfirmed_short = bool(
        cfg.get("backtest", {}).get("allow_unconfirmed_short_in_bt", False)
    )
    risk_cfg.allow_short_without_confirmed_shortability = allow_unconfirmed_short

    if allow_unconfirmed_short:
        print("[WARN] shortability未確認売りを許可しています。")

    if not cfg.get("backtest", {}).get("impact_k_is_calibrated", False):
        print("[WARN] impact_k_bpは未較正です。")

    bt_start = pd.Timestamp(cfg["backtest"]["start"]).date()
    bt_end = pd.Timestamp(cfg["backtest"]["end"]).date()

    lookback = int(cfg.get("model", {}).get("lookback", 5))
    top_n = int(cfg.get("model", {}).get("top_n", 5))
    adv_window = int(
        cfg["backtest"].get(
            "adv_window",
            cfg["sizing"].get("adv_window", 20),
        )
    )
    min_adv_periods = int(
        cfg["backtest"].get(
            "min_adv_periods",
            cfg["sizing"].get("min_adv_periods", 20),
        )
    )
    holding_days = int(cfg["backtest"].get("holding_days", 1))

    warmup_days = max(lookback + 1, adv_window, min_adv_periods) * 3
    load_start = bt_start - timedelta(days=warmup_days)

    with Storage(cfg["data"]["db_path"]) as storage:
        universe_all = load_universe(cfg["universe"])
        codes = universe_all["code"].astype(str).tolist()

        prices = storage.load_prices(
            codes,
            load_start.isoformat(),
            bt_end.isoformat(),
        )
        shortability = storage.load_shortability(
            codes,
            load_start.isoformat(),
            bt_end.isoformat(),
        )

    if prices.empty:
        print("価格データが空です。")
        return

    prices = prices.copy()
    prices["_date_ts"] = pd.to_datetime(prices["date"])

    all_dates = sorted(pd.to_datetime(prices["date"]).strftime("%Y-%m-%d").unique())
    all_dates = [d for d in all_dates if cfg["backtest"]["start"] <= d <= cfg["backtest"]["end"]]

    if not all_dates:
        print("BT期間に該当する日付がありません。")
        return

    model = MeanReversionRule(lookback=lookback, top_n=top_n)

    dq_cfg = cfg.get("data_quality", {})
    thresholds = CoverageThresholds.from_config(dq_cfg)

    all_order_frames: list[pd.DataFrame] = []

    for d in all_dates:
        univ_d = load_universe(cfg["universe"], as_of=d)
        codes_d = set(univ_d["code"].tolist())
        px_d = prices[prices["code"].isin(codes_d)]
        if px_d.empty:
            continue

        report = validate_daily_coverage(
            px_d,
            univ_d,
            as_of=d,
            lookback=lookback,
            adv_window=adv_window,
            min_adv_periods=min_adv_periods,
            thresholds=thresholds,
        )
        if not report.ok:
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
            shortability=shortability if not shortability.empty else None,
            universe=univ_d,
            holding_days=holding_days,
            order_type="MKT_OPEN",
            unit=int(cfg.get("sizing", {}).get("unit", 100)),
            for_backtest=True,
            shortability_max_age_days=int(
                cfg.get("risk", {}).get(
                    "shortability_max_age_days",
                    4,
                )
            ),
        )
        if not day_orders.empty:
            all_order_frames.append(day_orders)

    if not all_order_frames:
        print("シグナル0件。バックテストをスキップします。")
        return

    all_orders = pd.concat(all_order_frames, ignore_index=True)

    bt = PortfolioBacktester(
        initial_capital=float(cfg["backtest"].get("initial_capital", 100_000_000)),
        risk=risk_cfg,
        impact_k_bp=float(cfg["backtest"].get("impact_k_bp", 30.0)),
        annual_interest_rate=float(cfg["backtest"].get("annual_interest_rate", 0.02)),
        annual_lending_rate=float(cfg["backtest"].get("short_lending_rate", 0.02)),
        commission_bp=float(cfg["backtest"].get("commission_bp", 0.0)),
        half_spread_bp=float(cfg["backtest"].get("half_spread_bp", 0.0)),
        adv_window=adv_window,
        min_adv_periods=min_adv_periods,
        require_liquidity_data=True,
    )

    result = bt.run(
        all_orders,
        prices,
        start_date=cfg["backtest"]["start"],
        end_date=cfg["backtest"]["end"],
    )

    if result.trades.empty:
        print("約定0件。")
        return

    summary = summarize_portfolio_ledger(
        result.daily_ledger,
        initial_capital=float(cfg["backtest"].get("initial_capital", 100_000_000)),
        risk_free_rate=float(cfg["backtest"].get("risk_free_rate", 0.0)),
    )

    print(f"約定: {len(result.trades)}")
    print(f"リジェクト: {len(result.rejected_orders)}")
    print(f"最終NAV: {summary['final_nav']:,.0f}円")
    print(f"総リターン: {summary['total_return'] * 100:.2f}%")
    print(f"Sharpe: {summary['sharpe']:.2f}")
    print(f"最大DD: {summary['max_drawdown_pct'] * 100:.2f}%")
    print(f"最大グロス: {summary['max_gross_exposure']:,.0f}円")
    print(f"平均グロス: {summary['average_gross_exposure']:,.0f}円")

    out_dir = cfg.get("backtest", {}).get("output_dir", "./data/bt_out")
    os.makedirs(out_dir, exist_ok=True)

    result.trades.to_csv(os.path.join(out_dir, "portfolio_trades.csv"), index=False)
    result.daily_ledger.to_csv(os.path.join(out_dir, "daily_ledger.csv"), index=False)
    result.rejected_orders.to_csv(os.path.join(out_dir, "rejected_orders.csv"), index=False)

    write_backtest_manifest(
        out_dir=out_dir,
        config=cfg,
        config_path=args.config,
        prices=prices,
        universe=universe_all,
        shortability=shortability if not shortability.empty else None,
        extra={"summary": _json_safe_summary(summary)},
    )

    print(f"出力先: {out_dir}/")


if __name__ == "__main__":
    main()
