"""日次パイプラインのエントリポイント。

平日 08:15 に cron / タスクスケジューラから `python main.py` を実行する想定。
python main.py --dry-run で通知のみテスト（DB書込/発注指示送信を実行）。

プロセスロックにより cron の重複実行を防止する。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from jp_signal.pipeline import load_config, morning_pipeline


def _try_lock() -> str | None:
    """プロセスロックファイルを作成する。既存なら None を返す。"""
    import os

    lock_path = "/tmp/jp_signal.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return lock_path
    except FileExistsError:
        return None


def _release_lock(lock_path: str | None) -> None:
    if lock_path:
        import os

        try:
            os.unlink(lock_path)
        except OSError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日次パイプライン")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="通知のみテスト（DB書込/発注指示送信を実行しない）",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="基準日（YYYY-MM-DD）。未指定なら本日",
    )
    args = parser.parse_args()

    lock_path = _try_lock()
    if lock_path is None:
        print("[main] プロセスロック取得失敗: 既に実行中の可能性あり", file=sys.stderr)
        sys.exit(1)

    try:
        as_of = date.fromisoformat(args.date) if args.date else date.today()
        cfg = load_config()
        morning_pipeline(as_of, cfg, dry_run=args.dry_run)
    finally:
        _release_lock(lock_path)
