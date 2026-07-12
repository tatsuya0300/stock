#!/usr/bin/env python3
"""推奨バックテスト実行スクリプト。

config.yaml から設定を読み込み、PortfolioBacktester を用いて
バックテストを実行する。

Usage:
    python scripts/run_portfolio_backtest.py [--config CONFIG_PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from jp_signal.config import load_config, ConfigError
from jp_signal.storage import Storage
from jp_signal.portfolio import PortfolioBacktester

log = logging.getLogger(__name__)


def _decision_at_jst(
    bt_end: str,
    cfg: dict,
) -> pd.Timestamp:
    """バックテスト終了日の JST 営業日終了時刻を返す。

    各日の注文生成時点のタイムスタンプとして使用する。
    point_in_time モードでは、この時刻までに利用可能になった
    price revision のみを参照する。
    """
    # デフォルトは当日 15:00 JST (= 06:00 UTC)
    end_ts = pd.Timestamp(bt_end)

    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("Asia/Tokyo")

    return end_ts.replace(hour=15, minute=0, second=0, microsecond=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run portfolio backtest")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- config ---
    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        log.error("Config error: %s", e)
        sys.exit(1)

    # --- storage ---
    db_path = cfg["data"]["db_path"]
    storage = Storage(db_path, read_only=True)

    # --- parameters ---
    bt_cfg = cfg["backtest"]
    start_date = str(bt_cfg["start"])
    end_date = str(bt_cfg["end"])
    initial_capital = float(bt_cfg.get("initial_capital", 100_000_000))

    risk_cfg = cfg.get("risk", {})
    from jp_signal.risk import risk_config_from_dict
    risk = risk_config_from_dict(risk_cfg)

    impact_k_bp = float(bt_cfg.get("impact_k_bp", 30.0))
    annual_interest_rate = float(bt_cfg.get("annual_interest_rate", 0.02))
    annual_lending_rate = float(bt_cfg.get("short_lending_rate", 0.02))
    commission_bp = float(bt_cfg.get("commission_bp", 0.0))
    half_spread_bp = float(bt_cfg.get("half_spread_bp", 0.0))
    adv_window = int(bt_cfg.get("adv_window", 20))
    min_adv_periods = int(bt_cfg.get("min_adv_periods", 20))
    require_liquidity = bool(bt_cfg.get("require_liquidity_data", True))
    maintain_margin_ratio = float(bt_cfg.get("maintain_margin_ratio", 0.25))

    data_cfg = cfg["data"]
    vintage_mode = str(data_cfg.get("price_vintage_mode", "latest_snapshot"))

    # --- load orders ---
    log.info("Loading orders...")
    orders = storage.load_orders(start=start_date, end=end_date)
    if orders.empty:
        log.warning("No orders found for period %s - %s", start_date, end_date)
    else:
        log.info("Loaded %d orders", len(orders))

    # --- load prices ---
    codes: list[str] = []
    if not orders.empty:
        codes = sorted(orders["code"].astype(str).unique().tolist())

    load_start = pd.Timestamp(start_date) - pd.Timedelta(days=adv_window + 10)

    log.info(
        "Loading prices (mode=%s, codes=%d, range=%s to %s)...",
        vintage_mode,
        len(codes),
        load_start.date(),
        end_date,
    )

    if vintage_mode == "point_in_time":
        # BT終了日時点で利用可能だったリビジョンで一括取得
        decision_ts = _decision_at_jst(end_date, cfg)
        prices = storage.load_prices_for_backtest(
            codes=codes,
            start=load_start.isoformat(),
            end=end_date,
            decision_at=decision_ts.tz_convert("UTC").isoformat(),
            vintage_mode=vintage_mode,
        )
    else:
        prices = storage.load_prices(
            codes,
            load_start.isoformat(),
            end_date,
        )

    log.info("Loaded %d price rows", len(prices))

    # --- run backtest ---
    log.info("Running backtest %s -> %s ...", start_date, end_date)

    bt = PortfolioBacktester(
        initial_capital=initial_capital,
        risk=risk,
        impact_k_bp=impact_k_bp,
        annual_interest_rate=annual_interest_rate,
        annual_lending_rate=annual_lending_rate,
        commission_bp=commission_bp,
        half_spread_bp=half_spread_bp,
        adv_window=adv_window,
        min_adv_periods=min_adv_periods,
        require_liquidity_data=require_liquidity,
        maintain_margin_ratio=maintain_margin_ratio,
    )

    if vintage_mode == "point_in_time" and not orders.empty:
        # --- 日次 point-in-time で注文生成 ---
        # 各取引日ごとに、その日時点で利用可能だった価格 revision で
        # prices を絞り込んで BT を実行する
        trading_dates = sorted(prices["date"].unique())
        all_trades: list[pd.DataFrame] = []
        all_rejected: list[pd.DataFrame] = []
        all_ledger: list[pd.DataFrame] = []
        all_open: list[pd.DataFrame] = []
        all_ca: list[pd.DataFrame] = []

        for i, d in enumerate(trading_dates):
            day_str = str(pd.Timestamp(d).date())
            log.debug("PIT step: %s", day_str)

            day_orders = orders[orders["order_date"] == day_str]
            if day_orders.empty:
                continue

            # 当日時点のPIT価格を取得
            decision_ts = _decision_at_jst(day_str, cfg)
            day_codes: list[str] = (
                day_orders["code"].astype(str).unique().tolist()
            )

            day_px = storage.load_prices_asof(
                asof_date=decision_ts.tz_convert("UTC").isoformat(),
                codes=day_codes,
                start=load_start.isoformat(),
                end=day_str,
            )

            if day_px.empty:
                log.warning("No PIT prices for %s, skipping", day_str)
                continue

            # 全履歴価格とマージ（当日以前の全ての価格が必要）
            full_px = prices[
                (prices["code"].isin(day_codes))
                & (prices["date"] <= day_str)
            ]

            # PIT価格で上書き
            combined = full_px.copy()
            for _, pit_row in day_px.iterrows():
                mask = (
                    (combined["code"] == pit_row["code"])
                    & (combined["date"] == pit_row["date"])
                )
                if mask.any():
                    for col in [
                        "open", "high", "low", "close",
                        "adj_open", "adj_high", "adj_low", "adj_close",
                        "volume", "turnover",
                    ]:
                        combined.loc[mask, col] = pit_row[col]
                else:
                    combined = pd.concat(
                        [combined, pit_row.to_frame().T], ignore_index=True
                    )

            if i == 0:
                # 最初の日のみフルBT（初期状態から）
                result = bt.run(
                    day_orders,
                    combined,
                    start_date=day_str,
                    end_date=day_str,
                )
            else:
                result = bt.run(
                    day_orders,
                    combined,
                    start_date=day_str,
                    end_date=day_str,
                )

            if result.trades is not None and not result.trades.empty:
                all_trades.append(result.trades)
            if result.rejected_orders is not None and not result.rejected_orders.empty:
                all_rejected.append(result.rejected_orders)
            if result.daily_ledger is not None and not result.daily_ledger.empty:
                all_ledger.append(result.daily_ledger)
            if result.open_positions is not None and not result.open_positions.empty:
                all_open.append(result.open_positions)
            if result.corporate_action_events is not None and not result.corporate_action_events.empty:
                all_ca.append(result.corporate_action_events)

        result_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        result_rejected = pd.concat(all_rejected, ignore_index=True) if all_rejected else pd.DataFrame()
        result_ledger = pd.concat(all_ledger, ignore_index=True) if all_ledger else pd.DataFrame()
        result_open = pd.concat(all_open, ignore_index=True) if all_open else pd.DataFrame()
        result_ca = pd.concat(all_ca, ignore_index=True) if all_ca else pd.DataFrame()

        from jp_signal.portfolio import PortfolioResult
        result = PortfolioResult(
            trades=result_trades,
            rejected_orders=result_rejected,
            daily_ledger=result_ledger,
            open_positions=result_open,
            corporate_action_events=result_ca,
        )
    else:
        # --- 一括実行（latest_snapshot モード） ---
        result = bt.run(
            orders,
            prices,
            start_date=start_date,
            end_date=end_date,
        )

    # --- output ---
    out_dir = Path(cfg.get("output", {}).get("dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.trades is not None and not result.trades.empty:
        trades_path = out_dir / "trades.csv"
        result.trades.to_csv(trades_path, index=False)
        log.info("Trades saved: %s (%d rows)", trades_path, len(result.trades))

    if result.rejected_orders is not None and not result.rejected_orders.empty:
        rejected_path = out_dir / "rejected_orders.csv"
        result.rejected_orders.to_csv(rejected_path, index=False)
        log.info("Rejected orders saved: %s (%d rows)", rejected_path, len(result.rejected_orders))

    if result.daily_ledger is not None and not result.daily_ledger.empty:
        ledger_path = out_dir / "daily_ledger.csv"
        result.daily_ledger.to_csv(ledger_path, index=False)
        log.info("Daily ledger saved: %s (%d rows)", ledger_path, len(result.daily_ledger))

    if result.open_positions is not None and not result.open_positions.empty:
        open_path = out_dir / "open_positions.csv"
        result.open_positions.to_csv(open_path, index=False)
        log.info("Open positions saved: %s (%d rows)", open_path, len(result.open_positions))

    if result.corporate_action_events is not None and not result.corporate_action_events.empty:
        ca_path = out_dir / "corporate_action_events.csv"
        result.corporate_action_events.to_csv(ca_path, index=False)
        log.info("CA events saved: %s (%d rows)", ca_path, len(result.corporate_action_events))

    # --- summary ---
    if result.daily_ledger is not None and not result.daily_ledger.empty:
        final_nav = result.daily_ledger.iloc[-1]["nav"]
        total_return = (final_nav / initial_capital - 1) * 100
        n_trades = len(result.trades) if result.trades is not None else 0
        log.info(
            "Backtest complete: NAV=%.2f return=%.2f%% trades=%d",
            final_nav,
            total_return,
            n_trades,
        )

    storage.close()


if __name__ == "__main__":
    main()
