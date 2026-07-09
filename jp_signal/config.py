"""設定管理（FR-CONFIG-01/02）。

config.yaml の読み込みとバリデーションを行う。
秘密情報は環境変数で上書き可能:
  - JQUANTS_API_KEY（V2。推奨）
  - JQUANTS_REFRESH_TOKEN（V1。後方互換）
  - DISCORD_WEBHOOK
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


REQUIRED_TOP_LEVEL_KEYS = {"data", "universe", "backtest", "sizing", "notify"}

_DEFAULT_RISK = {
    "max_orders_per_day": 10,
    "max_gross_exposure_yen": 100_000_000.0,
    "max_single_name_exposure_yen": 20_000_000.0,
    "max_long_exposure_yen": 100_000_000.0,
    "max_short_exposure_yen": 100_000_000.0,
    "allow_short_without_confirmed_shortability": False,
}


def load_config(path: str = "config.yaml") -> dict:
    """config.yaml を読み込みバリデーションする。

    環境変数 JQUANTS_API_KEY または JQUANTS_REFRESH_TOKEN / DISCORD_WEBHOOK で
    config.yaml の値を上書きできる。
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

    missing = REQUIRED_TOP_LEVEL_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(f"設定ファイルに必須キーが不足: {sorted(missing)}")

    # 環境変数で秘密情報を上書き
    # J-Quants V2: API Key
    env_api_key = os.getenv("JQUANTS_API_KEY")
    if env_api_key:
        cfg.setdefault("data", {})["jquants_api_key"] = env_api_key

    # J-Quants V1: Refresh Token（後方互換）
    env_token = os.getenv("JQUANTS_REFRESH_TOKEN")
    if env_token:
        cfg.setdefault("data", {})["jquants_refresh_token"] = env_token

    env_webhook = os.getenv("DISCORD_WEBHOOK")
    if env_webhook:
        cfg.setdefault("notify", {})["discord_webhook"] = env_webhook

    # data.source のバリデーション
    valid_sources = {"yfinance", "jquants"}
    source = cfg.get("data", {}).get("source", "")
    if source not in valid_sources:
        raise ValueError(
            f"data.source は {valid_sources} のいずれかである必要があります "
            f"(actual: {source!r})"
        )

    if source == "jquants":
        api_key = cfg.get("data", {}).get("jquants_api_key", "")
        if not api_key:
            raise ValueError(
                "data.source=jquants の場合は環境変数 JQUANTS_API_KEY "
                "(V2 API Key) を設定してください。"
            )

    # sizing のバリデーション
    sizing = cfg.get("sizing", {})
    adv_ratio = float(sizing.get("adv_ratio", 0.001))
    adv_ratio_cap = float(sizing.get("adv_ratio_cap", 0.002))
    if adv_ratio <= 0:
        raise ValueError(
            f"sizing.adv_ratio ({adv_ratio}) は 0 より大きい必要があります"
        )
    if adv_ratio > adv_ratio_cap:
        raise ValueError(
            f"sizing.adv_ratio ({adv_ratio}) が "
            f"sizing.adv_ratio_cap ({adv_ratio_cap}) を超えています"
        )

    # notify.channel のバリデーション
    valid_channels = {"console", "discord"}
    channel = cfg.get("notify", {}).get("channel", "console")
    if channel not in valid_channels:
        raise ValueError(
            f"notify.channel は {valid_channels} のいずれかである必要があります"
        )

    if channel == "discord":
        webhook = cfg.get("notify", {}).get("discord_webhook", "")
        if not webhook:
            raise ValueError(
                "discord チャンネル利用時は notify.discord_webhook か "
                "環境変数 DISCORD_WEBHOOK が必要です"
            )

    # risk デフォルト補完
    cfg.setdefault("risk", {})
    for k, v in _DEFAULT_RISK.items():
        cfg["risk"].setdefault(k, v)

    return cfg
