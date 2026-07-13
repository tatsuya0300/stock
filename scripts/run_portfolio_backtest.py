#!/usr/bin/env python3
"""ポートフォリオバックテスト。

Modes:
    research:
        各日の判断時点で利用可能だった価格から注文を再生成する。

    replay:
        DBに保存された実際のordersを再生する。

PIT設計:
    - シグナル生成価格はdecision_at以前に利用可能だったrevisionのみ。
    - 約定価格は当日の実際のraw OHLCを使用する。
    - 全日分の注文を作った後、PortfolioBacktester.run()を1回だけ呼ぶ。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from jp_signal.config import ConfigError, load_config
from jp_signal.corporate_actions import (
    prepare_corporate_actions,
)
from jp_signal.historical_orders import (
    generate_historical_orders,
)
from jp_signal.model import model_from_config
from jp_signal.portfolio import PortfolioBacktester
from jp_signal.portfolio_metrics import (
    summarize_portfolio_ledger,
)
from jp_signal.risk import risk_config_from_dict
from jp_signal.storage import Storage
from jp_signal.universe import load_universe

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run portfolio backtest"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["research", "replay"],
        default="research",
        help=(
            "research=過去価格から注文を再生成、"
            "replay=DB保存済み注文を再生"
        ),
    )
    return parser.parse_args()


def _decision_at_jst(
    trading_date: date | str,
    cfg: dict,
) -> pd.Timestamp:
    raw_time = str(
        cfg.get("notify", {}).get(
            "morning_time",
            "08:15",
        )
    )

    timestamp = pd.Timestamp(
        f"{pd.Timestamp(trading_date).date()} "
        f"{raw_time}"
    )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "Asia/Tokyo"
        )

    return timestamp


def _model_factory(
    model_cfg: dict,
):
    return model_from_config(model_cfg)


def _load_corporate_actions(
    cfg: dict,
):
    bt_cfg = cfg.get("backtest", {})
    path = bt_cfg.get(
        "corporate_actions_file"
    )
    required = bool(
        bt_cfg.get(
            "require_corporate_actions",
            False,
        )
    )

    if not path:
        if required:
            raise ConfigError(
                "require_corporate_actions=true "
                "ですがcorporate_actions_fileがありません"
            )
        return []

    csv_path = Path(path)

    if not csv_path.exists():
        if required:
            raise FileNotFoundError(
                f"corporate actions not found: "
                f"{csv_path}"
            )

        log.warning(
            "corporate actions file not found: %s",
            csv_path,
        )
        return []

    frame = pd.read_csv(
        csv_path,
        dtype={"code": str},
    )

    return prepare_corporate_actions(frame)


def _write_result(
    result,
    cfg: dict,
    initial_capital: float,
) -> None:
    output_dir = Path(
        cfg["backtest"].get(
            "output_dir",
            "./data/bt_out",
        )
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "portfolio_trades.csv": result.trades,
        "rejected_orders.csv": (
            result.rejected_orders
        ),
        "daily_ledger.csv": (
            result.daily_ledger
        ),
        "open_positions.csv": (
            result.open_positions
        ),
        "corporate_action_events.csv": (
            result.corporate_action_events
        ),
    }

    for filename, frame in outputs.items():
        if frame is None or frame.empty:
            continue

        path = output_dir / filename
        frame.to_csv(path, index=False)
        log.info(
            "wrote %s rows=%d",
            path,
            len(frame),
        )

    if (
        result.daily_ledger is not None
        and not result.daily_ledger.empty
    ):
        summary = summarize_portfolio_ledger(
            result.daily_ledger,
            initial_capital=initial_capital,
            risk_free_rate=float(
                cfg["backtest"].get(
                    "risk_free_rate",
                    0.0,
                )
            ),
        )

        serializable = {
            key: value
            for key, value in summary.items()
            if key
            not in {
                "daily_returns",
                "equity_curve",
            }
        }

        pd.Series(
            serializable,
            name="value",
        ).to_csv(
            output_dir / "summary.csv",
            header=True,
        )


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s [%(levelname)s] "
            "%(name)s: %(message)s"
        ),
    )

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        log.error(
            "config load failed: %s",
            exc,
        )
        raise

    bt_cfg = cfg["backtest"]
    sizing_cfg = cfg["sizing"]
    risk_dict = dict(
        cfg.get("risk", {})
    )

    start_date = pd.Timestamp(
        bt_cfg["start"]
    ).date()
    end_date = pd.Timestamp(
        bt_cfg["end"]
    ).date()

    if start_date > end_date:
        raise ConfigError(
            "backtest.start must be <= end"
        )

    initial_capital = float(
        bt_cfg.get(
            "initial_capital",
            100_000_000,
        )
    )

    risk = risk_config_from_dict(
        risk_dict
    )

    db_path = cfg["data"]["db_path"]

    with Storage(
        db_path,
        read_only=True,
    ) as storage:
        all_universe = load_universe(
            cfg["universe"]
        )
        codes = (
            all_universe["code"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        adv_window = int(
            bt_cfg.get(
                "adv_window",
                20,
            )
        )
        model_lookback = int(
            cfg.get("model", {}).get(
                "lookback",
                5,
            )
        )

        warmup_days = (
            max(
                adv_window,
                model_lookback,
            )
            * 3
        )

        load_start = (
            start_date
            - timedelta(
                days=warmup_days
            )
        )

        # 執行価格は最終的に確定したraw OHLCを使う。
        execution_prices = storage.load_prices(
            codes,
            load_start.isoformat(),
            end_date.isoformat(),
        )

        if execution_prices.empty:
            raise RuntimeError(
                "execution prices are empty"
            )

        execution_prices["date"] = (
            pd.to_datetime(
                execution_prices["date"]
            ).dt.strftime("%Y-%m-%d")
        )

        trading_dates = sorted(
            {
                pd.Timestamp(value).date()
                for value in execution_prices["date"]
                if (
                    start_date
                    <= pd.Timestamp(value).date()
                    <= end_date
                )
            }
        )

        if not trading_dates:
            raise RuntimeError(
                "no trading dates in test window"
            )

        if args.mode == "replay":
            orders = storage.load_orders(
                start=start_date.isoformat(),
                end=end_date.isoformat(),
            )

            if not orders.empty:
                orders = orders.rename(
                    columns={
                        "order_date": "date",
                    }
                )
                orders["holding_days"] = int(
                    bt_cfg.get(
                        "holding_days",
                        1,
                    )
                )

        else:
            vintage_mode = str(
                cfg["data"].get(
                    "price_vintage_mode",
                    "latest_snapshot",
                )
            )

            def price_loader(
                requested_codes: list[str],
                history_start: date,
                decision_date: date,
            ) -> pd.DataFrame:
                if (
                    vintage_mode
                    == "point_in_time"
                ):
                    decision_at = (
                        _decision_at_jst(
                            decision_date,
                            cfg,
                        )
                    )

                    return storage.load_prices_asof(
                        asof_date=(
                            decision_at
                            .tz_convert("UTC")
                            .isoformat()
                        ),
                        codes=requested_codes,
                        start=(
                            history_start
                            .isoformat()
                        ),
                        end=(
                            decision_date
                            .isoformat()
                        ),
                    )

                return storage.load_prices(
                    requested_codes,
                    history_start.isoformat(),
                    decision_date.isoformat(),
                )

            def shortability_loader(
                requested_codes: list[str],
                decision_at: pd.Timestamp,
            ) -> pd.DataFrame:
                return (
                    storage
                    .load_shortability_observations(
                        requested_codes,
                        available_before=(
                            decision_at
                        ),
                        available_after=(
                            decision_at
                            - pd.Timedelta(
                                days=30
                            )
                        ),
                    )
                )

            historical = (
                generate_historical_orders(
                    trading_dates=trading_dates,
                    model_factory=(
                        _model_factory
                    ),
                    price_loader=(
                        price_loader
                    ),
                    shortability_loader=(
                        shortability_loader
                    ),
                    universe_loader=(
                        lambda as_of:
                        load_universe(
                            cfg["universe"],
                            as_of=as_of,
                        )
                    ),
                    model_cfg=cfg.get(
                        "model",
                        {},
                    ),
                    sizing_cfg=sizing_cfg,
                    risk_cfg=risk_dict,
                    holding_days=int(
                        bt_cfg.get(
                            "holding_days",
                            1,
                        )
                    ),
                    unit=int(
                        sizing_cfg.get(
                            "unit",
                            100,
                        )
                    ),
                    decision_time=str(
                        cfg.get(
                            "notify",
                            {},
                        ).get(
                            "morning_time",
                            "08:15",
                        )
                    ),
                    shortability_max_age_days=int(
                        risk_dict.get(
                            "shortability_max_age_days",
                            4,
                        )
                    ),
                    fail_fast=True,
                )
            )

            orders = historical.orders

            log.info(
                "historical order diagnostics: %s",
                historical.diagnostics,
            )

            if (
                historical.rejections
                is not None
                and not historical.rejections.empty
            ):
                rejection_output = Path(
                    bt_cfg.get(
                        "output_dir",
                        "./data/bt_out",
                    )
                )
                rejection_output.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                historical.rejections.to_csv(
                    rejection_output
                    / "order_build_rejections.csv",
                    index=False,
                )

        if orders is None:
            orders = pd.DataFrame()

        corporate_actions = (
            _load_corporate_actions(cfg)
        )

        backtester = PortfolioBacktester(
            initial_capital=initial_capital,
            risk=risk,
            impact_k_bp=float(
                bt_cfg.get(
                    "impact_k_bp",
                    30.0,
                )
            ),
            annual_interest_rate=float(
                bt_cfg.get(
                    "annual_interest_rate",
                    0.02,
                )
            ),
            annual_lending_rate=float(
                bt_cfg.get(
                    "short_lending_rate",
                    0.02,
                )
            ),
            commission_bp=float(
                bt_cfg.get(
                    "commission_bp",
                    0.0,
                )
            ),
            half_spread_bp=float(
                bt_cfg.get(
                    "half_spread_bp",
                    0.0,
                )
            ),
            adv_window=adv_window,
            min_adv_periods=int(
                bt_cfg.get(
                    "min_adv_periods",
                    20,
                )
            ),
            require_liquidity_data=bool(
                bt_cfg.get(
                    "require_liquidity_data",
                    True,
                )
            ),
            maintain_margin_ratio=float(
                bt_cfg.get(
                    "maintain_margin_ratio",
                    0.25,
                )
            ),
            account_type=str(
                bt_cfg.get(
                    "account_type",
                    "margin",
                )
            ),
        )

        # 重要：全期間を1回だけ実行する。
        result = backtester.run(
            orders,
            execution_prices,
            start_date=(
                start_date.isoformat()
            ),
            end_date=(
                end_date.isoformat()
            ),
            corporate_actions=(
                corporate_actions
            ),
        )

    _write_result(
        result,
        cfg,
        initial_capital,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception(
            "portfolio backtest failed"
        )
        sys.exit(1)
