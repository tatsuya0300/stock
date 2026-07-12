#!/usr/bin/env python3
"""fills テーブル v6 マイグレーション（PR-3）。

ALTER TABLE で fills テーブルに下記カラムを追加する:
  - exit_date          TEXT
  - exit_price         REAL
  - exit_slippage_bp   REAL
  - holding_days       INTEGER
  - carry_cost         REAL
  - carry_days         INTEGER
  - pnl                REAL

Usage:
    python scripts/migrate_fills_v6.py <db_path>
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def migrate_fills_v6(db_path: str | Path) -> int:
    """fills テーブルに v6 カラムを追加する。

    Returns:
        実行した ALTER TABLE の件数。
    """
    statements = [
        "ALTER TABLE fills ADD COLUMN exit_date TEXT DEFAULT NULL",
        "ALTER TABLE fills ADD COLUMN exit_price REAL DEFAULT NULL",
        "ALTER TABLE fills ADD COLUMN exit_slippage_bp REAL DEFAULT NULL",
        "ALTER TABLE fills ADD COLUMN holding_days INTEGER DEFAULT NULL",
        "ALTER TABLE fills ADD COLUMN carry_cost REAL DEFAULT 0.0",
        "ALTER TABLE fills ADD COLUMN carry_days INTEGER DEFAULT 0",
        "ALTER TABLE fills ADD COLUMN pnl REAL DEFAULT NULL",
    ]

    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"Database not found: {p}")

    conn = sqlite3.connect(str(p))
    cursor = conn.execute("PRAGMA table_info(fills)")
    existing = {row[1] for row in cursor.fetchall()}

    applied = 0
    for stmt in statements:
        col = stmt.split()[4]
        if col in existing:
            print(f"SKIP: column '{col}' already exists")
            continue
        conn.execute(stmt)
        print(f"OK: {stmt}")
        applied += 1

    conn.commit()
    conn.close()
    print(f"\nApplied {applied} column(s) to fills table.")
    return applied


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    db_path = sys.argv[1]
    migrate_fills_v6(db_path)


if __name__ == "__main__":
    main()
