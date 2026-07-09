"""日次パイプラインのエントリポイント。

平日 08:15 に cron / タスクスケジューラから python main.py を実行する想定。
"""

from datetime import date

from jp_signal.config import load_config
from jp_signal.pipeline import morning_pipeline

if __name__ == "__main__":
    cfg = load_config()
    morning_pipeline(date.today(), cfg)
