"""通知アダプタ（FR-NOTIFY-01〜06, FR-COMP）。

Notifier インターフェースで Console / Discord / Slack を差し替え可能にする。
コンプライアンス定型文を常時付与する（FR-COMP）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

COMPLIANCE_FOOTER = (
    "─────────────\n"
    "※本通知は投資助言ではなく、システムが生成した参考情報です。\n"
    "※最終的な投資判断はご自身の責任で行ってください。売買を推奨するものではありません。"
)

DISCORD_MAX_LENGTH = 2000


class Notifier(ABC):
    """通知送信インターフェース。"""

    @abstractmethod
    def send(self, title: str, body: str) -> None:
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    """標準出力への通知（MVPデフォルト）。"""

    def send(self, title: str, body: str) -> None:
        print(f"=== {title} ===\n{body}\n")


class DiscordNotifier(Notifier):
    """Discord Webhook への通知。長文は分割送信する。"""

    def __init__(self, webhook_url: str):
        self.url = webhook_url

    def send(self, title: str, body: str) -> None:
        import requests

        chunks = _split_text(body, DISCORD_MAX_LENGTH)
        for i, chunk in enumerate(chunks):
            t = f"{title} ({i+1}/{len(chunks)})" if len(chunks) > 1 else title
            r = requests.post(
                self.url,
                json={"content": f"**{t}**\n\n{chunk}"},
                timeout=15,
            )
            r.raise_for_status()


def _split_text(text: str, max_len: int) -> list[str]:
    """max_len を超えるテキストを改行単位で分割する。"""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            if current:
                chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


def format_orders(orders: pd.DataFrame) -> str:
    """FR-NOTIFY-06: 1銘柄1行の読みやすい書式で発注指示を整形する。"""
    if orders.empty:
        return f"注文なし\n\n{COMPLIANCE_FOOTER}"

    lines = []
    for _, o in orders.iterrows():
        side = "買" if o["side"] == "BUY" else "売"
        shortable = "売可" if o.get("shortable", True) else "売不可"
        warn = f" ⚠{o['warn']}" if o.get("warn") else ""
        lines.append(
            f"[{side}] {o['code']} {o.get('name', '')} "
            f"{o['order_type']} {int(o['qty'])}株 ¥{o['ref_price']:.0f} "
            f"{shortable}{warn}"
        )
    return "\n".join(lines) + "\n\n" + COMPLIANCE_FOOTER
