"""日次パイプラインのエントリポイント。

使い方:
  python main.py                  # 寄前（デフォルト）
  python main.py morning
  python main.py closing
  python main.py closing --fills data/fills_2026-07-09.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import date

from jp_signal.config import load_config
from jp_signal.pipeline import closing_pipeline, make_notifier, morning_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("main")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="jp_signal daily pipeline")
    p.add_argument(
        "mode",
        nargs="?",
        default="morning",
        choices=["morning", "closing"],
        help="morning=寄前, closing=引け後",
    )
    p.add_argument(
        "--fills",
        default=None,
        help="closing 時に取り込む fills CSV パス",
    )
    p.add_argument(
        "--date",
        default=None,
        help="処理日 YYYY-MM-DD（未指定なら今日）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="DB 書込をスキップ",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config()
    notifier = make_notifier(cfg)
    as_of = date.fromisoformat(args.date) if args.date else date.today()

    try:
        if args.mode == "morning":
            morning_pipeline(as_of, cfg, dry_run=args.dry_run)
        else:
            closing_pipeline(
                as_of,
                cfg,
                fills_csv=args.fills,
                dry_run=args.dry_run,
            )
    except Exception:
        log.exception("%s pipeline failed", args.mode)
        try:
            notifier.send(
                f"【障害】jp_signal {args.mode}_pipeline",
                traceback.format_exc()[-1500:],
            )
        except Exception:
            log.exception("failure notification also failed")
        raise
