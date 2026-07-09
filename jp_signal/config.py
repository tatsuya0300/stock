"""設定管理（FR-CONFIG-01/02）。

config.yaml の読み込みとバリデーションを行う。
"""

from __future__ import annotations

from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


REQUIRED_TOP_LEVEL_KEYS = {"data", "universe", "backtest", "sizing", "notify"}


def load_config(path: str = "config.yaml") -> dict:
    """config.yaml を読み込みバリデーションする。

    Args:
        path: YAML 設定ファイルのパス。

    Returns:
        バリデーション済みの設定 dict。

    Raises:
        FileNotFoundError: 設定ファイルが存在しない場合。
        ImportError: PyYAML がインストールされていない場合。
        ValueError: 設定値が不正な場合。
    """
    if yaml is None:
        raise ImportError("PyYAML が必要です: pip install pyyaml")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError("設定ファイルが空です")

    # トップレベルキーの確認
    missing = REQUIRED_TOP_LEVEL_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(f"設定ファイルに必須キーが不足: {sorted(missing)}")

    # data.source のバリデーション
    valid_sources = {"yfinance", "jquants"}
    source = cfg.get("data", {}).get("source", "")
    if source not in valid_sources:
        raise ValueError(
            f"data.source は {valid_sources} のいずれかである必要があります (actual: {source!r})"
        )

    # sizing のバリデーション
    sizing = cfg.get("sizing", {})
    adv_ratio = sizing.get("adv_ratio", 0.001)
    adv_ratio_cap = sizing.get("adv_ratio_cap", 0.002)
    if adv_ratio > adv_ratio_cap:
        raise ValueError(
            f"sizing.adv_ratio ({adv_ratio}) が sizing.adv_ratio_cap ({adv_ratio_cap}) を超えています"
        )

    # notify.channel のバリデーション
    valid_channels = {"console", "discord"}
    channel = cfg.get("notify", {}).get("channel", "console")
    if channel not in valid_channels:
        raise ValueError(
            f"notify.channel は {valid_channels} のいずれかである必要があります"
        )

    # discord の場合は webhook 必須
    if channel == "discord":
        webhook = cfg.get("notify", {}).get("discord_webhook", "")
        if not webhook:
            raise ValueError("discord チャンネル利用時は notify.discord_webhook が必要です")

    return cfg
