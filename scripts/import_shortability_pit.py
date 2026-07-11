#!/usr/bin/env python3
"""Import normalized point-in-time shortability observations into the database."""

from __future__ import annotations

import argparse

from jp_signal.shortability_pit import (
    load_shortability_observations_csv,
)
from jp_signal.storage import Storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import normalized point-in-time "
            "shortability observations"
        )
    )

    parser.add_argument(
        "csv",
        help="正規化shortability CSV",
    )
    parser.add_argument(
        "--db-path",
        default="./data/jp_signal.sqlite",
        help="SQLite database path",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = load_shortability_observations_csv(
        args.csv
    )

    with Storage(args.db_path) as storage:
        inserted = (
            storage.insert_shortability_observations(
                frame
            )
        )

    print(f"rows={len(frame)}")
    print(f"inserted={inserted}")
    print(f"duplicates={len(frame) - inserted}")


if __name__ == "__main__":
    main()
