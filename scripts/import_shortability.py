"""shortability CSV を DB へ取り込むスクリプト。

使い方:
    python scripts/import_shortability.py --csv data/shortability.csv
    python scripts/import_shortability.py --csv data/shortability.csv --source jsf_csv --db data/stock.db

CSV 想定列:
    code, is_margin_lendable, short_restricted [, date, effective_at, fetched_at]

注意:
    - 日証金（JSF）の CS フォーマットに依存する。フォーマット変更時は
      shortability_v2.load_shortability_csv() の修正が必要。
    - 本スクリプトは既存の shortability テーブルを更新し、
      shortability_observations に観測履歴を追加する。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jp_signal.shortability_v2 import load_shortability_csv
from jp_signal.storage import Storage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import shortability CSV into DB")
    p.add_argument(
        "--csv",
        required=True,
        help="shortability CSV ファイルのパス",
    )
    p.add_argument(
        "--db",
        default="data/stock.db",
        help="DB ファイルのパス（デフォルト: data/stock.db）",
    )
    p.add_argument(
        "--source",
        default="jsf_csv",
        help="データソース名（デフォルト: jsf_csv）",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.path.exists(args.csv):
        print(f"エラー: CSV ファイルが見つかりません: {args.csv}")
        sys.exit(1)

    print(f"shortability CSV を読み込み中: {args.csv}")
    try:
        df = load_shortability_csv(args.csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: CSV 読み込み失敗: {e}")
        sys.exit(1)

    if df.empty:
        print("警告: 読み込んだデータが空です。")
        return

    print(f"  行数: {len(df)}")
    print(f"  銘柄数: {df['code'].nunique()}")
    print(f"  日付範囲: {df['date'].min()} 〜 {df['date'].max()}")

    with Storage(args.db) as st:
        # shortability_observations に観測履歴を保存
        st.insert_shortability_observations(df, source=args.source)
        print(f"  shortability_observations へ {len(df)} 行を挿入しました")

        # shortability テーブルを更新（最新投影）
        short_simple = df[["code", "date", "is_margin_lendable", "short_restricted"]].drop_duplicates(
            subset=["code", "date"]
        )
        st.upsert_shortability(short_simple)
        print(f"  shortability テーブルへ {len(short_simple)} 行を upsert しました")

    print(f"完了: {args.csv} を取り込みました")


if __name__ == "__main__":
    main()
