#!/usr/bin/env python3
"""売建可否CSVをPIT形式でSQLiteへ取り込む。"""

from __future__ import annotations

import argparse

import pandas as pd

from jp_signal.config import load_config
from jp_signal.shortability_provider import (
    CsvShortabilityProvider,
)
from jp_signal.storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.yaml",
    )
    parser.add_argument(
        "--csv",
        required=True,
    )
    parser.add_argument(
        "--source",
        default="manual_csv",
    )
    parser.add_argument(
        "--fetched-at",
        default=None,
        help=(
            "ISO8601。未指定なら現在時刻。"
        ),
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.fetched_at:
        fetched_at = pd.Timestamp(
            args.fetched_at
        )
    else:
        fetched_at = pd.Timestamp.now(
            tz="Asia/Tokyo"
        )

    provider = CsvShortabilityProvider(
        args.csv,
        default_source=args.source,
    )

    observations = provider.fetch(
        fetched_at=fetched_at
    )

    with Storage(
        cfg["data"]["db_path"]
    ) as storage:
        inserted = (
            storage
            .insert_shortability_observations(
                observations
            )
        )

    print(
        f"rows={len(observations)} "
        f"inserted={inserted}"
    )


if __name__ == "__main__":
    main()
