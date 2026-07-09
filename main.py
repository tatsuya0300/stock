"""日次パイプラインのエントリポイント。

平日 08:15 に cron / タスクスケジューラから python main.py を実行する想定。
"""

from __future__ import annotations

import logging
import sys
import traceback
from datetime import date

from jp_signal.config import load_config
from jp_signal.notifier import ConsoleNotifier, DiscordNotifier
from jp_signal.pipeline import morning_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("main")


def _make_notifier(cfg: dict):
    if cfg.get("notify", {}).get("channel") == "discord":
        return DiscordNotifier(cfg["notify"]["discord_webhook"])
    return ConsoleNotifier()


if __name__ == "__main__":
    cfg = load_config()
    notifier = _make_notifier(cfg)
    try:
        morning_pipeline(date.today(), cfg)
    except Exception:
        log.exception("morning_pipeline failed")
        try:
            notifier.send(
                "【障害】jp_signal morning_pipeline",
                traceback.format_exc()[-1500:],
            )
        except Exception:
            log.exception("failure notification also failed")
        raise
