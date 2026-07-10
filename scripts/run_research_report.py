"""Portfolio backtest outputsから研究用reportを作る。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from jp_signal.research_metrics import (
    estimate_trade_turnover,
    summarize_capacity,
    summarize_research_performance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate research evaluation report")
    parser.add_argument(
        "--ledger",
        default="./data/bt_out/daily_ledger.csv",
    )
    parser.add_argument(
        "--trades",
        default="./data/bt_out/portfolio_trades.csv",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help=("CSV with date and return columns. TOPIX total-return series is preferred."),
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000_000,
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--output",
        default="./data/bt_out/research_report.json",
    )
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, pd.Series):
        return {str(index): float(item) for index, item in value.items()}

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}

    if hasattr(value, "item"):
        return value.item()

    return value


def main() -> None:
    args = parse_args()

    ledger = pd.read_csv(args.ledger)
    trades = pd.read_csv(args.trades)

    benchmark_returns = None

    if args.benchmark:
        benchmark = pd.read_csv(args.benchmark)

        required = {"date", "return"}
        missing = required - set(benchmark.columns)

        if missing:
            raise ValueError(f"benchmark CSV missing columns: {sorted(missing)}")

        benchmark["date"] = pd.to_datetime(benchmark["date"])

        benchmark_returns = (
            benchmark.drop_duplicates("date", keep="last")
            .set_index("date")["return"]
            .astype(float)
            .sort_index()
        )

    performance = summarize_research_performance(
        ledger,
        initial_capital=args.initial_capital,
        risk_free_rate=args.risk_free_rate,
        benchmark_returns=benchmark_returns,
    )

    turnover = estimate_trade_turnover(
        trades,
        ledger,
        initial_capital=args.initial_capital,
    )

    capacity = summarize_capacity(trades)

    report = {
        "performance": performance,
        "capacity": capacity,
        "turnover": {
            "average_daily_turnover": (
                float(turnover["turnover"].mean()) if not turnover.empty else 0.0
            ),
            "maximum_daily_turnover": (
                float(turnover["turnover"].max()) if not turnover.empty else 0.0
            ),
            "total_traded_notional": (
                float(turnover["traded_notional"].sum()) if not turnover.empty else 0.0
            ),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            json_safe(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
