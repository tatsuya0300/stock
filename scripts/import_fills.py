"""fills CSV を DB に取り込む最小スクリプト。

使い方:
  python scripts/import_fills.py data/fills.csv

CSV 列:
  trade_date,code,side,qty,price[,note,run_id]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jp_signal.config import load_config
from jp_signal.storage import Storage


def main() -> None:
    if len(sys.argv) < 2:
        print("使い方: python scripts/import_fills.py <fills.csv>")
        sys.exit(1)
    path = sys.argv[1]
    cfg = load_config()
    with Storage(cfg["data"]["db_path"]) as st:
        n = st.import_fills_csv(path)
        print(f"imported fills: {n}")


if __name__ == "__main__":
    main()
