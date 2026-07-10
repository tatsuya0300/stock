"""設定管理（FR-CONFIG-01/02）。

config.yaml の読み込みとバリデーションを行う。
秘密情報は環境変数で上書き可能:
  - JQUANTS_API_KEY（V2。推奨）
  - JQUANTS_REFRESH_TOKEN（V1。後方互換）
  - DISCORD_WEBHOOK
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

REQUIRED_TOP_LEVEL_KEYS = {"data", "universe", "backtest", "sizing", "notify"}

_DEFAULT_RISK = {
    "max_orders_per_day": 10,
    "max_gross_exposure_yen": 100_000_000.0,
    "max_single_name_exposure_yen": 20_000_000.0,
    "max_long_exposure_yen": 100_000_000.0,
    "max_short_exposure_yen": 100_000_000.0,
    "allow_short_without_confirmed_shortability": False,
}

_DEFAULT_DATA = {
    # yfinance 近似 turnover を sizing/impact に使うことを明示許可するか
    "allow_approximate_turnover": False,
}

_DEFAULT_BACKTEST = {
    "impact_k_is_calibrated": False,
    "allow_unconfirmed_short_in_bt": False,
    "min_adv_periods": 20,
}


class ConfigError(ValueError):
    """設定不備。"""


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
        raise ConfigError("設定ファイルが空です")

    missing = REQUIRED_TOP_LEVEL_KEYS - set(cfg.keys())
    if missing:
        raise ConfigError(f"設定ファイルに必須キーが不足: {sorted(missing)}")

    # デフォルト補完
    cfg.setdefault("data", {})
    for k, v in _DEFAULT_DATA.items():
        cfg["data"].setdefault(k, v)

    cfg.setdefault("backtest", {})
    for k, v in _DEFAULT_BACKTEST.items():
        cfg["backtest"].setdefault(k, v)

    cfg.setdefault("risk", {})
    for k, v in _DEFAULT_RISK.items():
        cfg["risk"].setdefault(k, v)

    # 環境変数で秘密情報を上書き
    env_api_key = os.getenv("JQUANTS_API_KEY")
    if env_api_key:
        cfg["data"]["jquants_api_key"] = env_api_key

    env_token = os.getenv("JQUANTS_REFRESH_TOKEN")
    if env_token:
        cfg["data"]["jquants_refresh_token"] = env_token

    env_webhook = os.getenv("DISCORD_WEBHOOK")
    if env_webhook:
        cfg.setdefault("notify", {})["discord_webhook"] = env_webhook

    # data.source のバリデーション
    valid_sources = {"yfinance", "jquants"}
    source = cfg.get("data", {}).get("source", "")
    if source not in valid_sources:
        raise ConfigError(
            f"data.source は {valid_sources} のいずれかである必要があります (actual: {source!r})"
        )

    if source == "jquants":
        api_key = cfg.get("data", {}).get("jquants_api_key", "")
        if not api_key:
            raise ConfigError(
                "data.source=jquants の場合は環境変数 JQUANTS_API_KEY "
                "(V2 API Key) を設定してください。"
            )

    # yfinance 近似ガード
    allow_approx = bool(cfg["data"].get("allow_approximate_turnover", False))
    if source == "yfinance" and not allow_approx:
        log.warning(
            "data.source=yfinance かつ allow_approximate_turnover=false: "
            "sizing/impact を使う処理は guard_approximate_turnover() で拒否されます。"
            "疎通確認のみなら allow_approximate_turnover=true を明示してください（非推奨）。"
        )

    # universe.file の存在確認（起動時に分かりやすくする）
    univ_file = cfg.get("universe", {}).get("file")
    if not univ_file:
        raise ConfigError("universe.file が設定されていません")
    if not Path(univ_file).exists():
        raise ConfigError(
            f"universe.file が見つかりません: {univ_file}\n"
            "  - 動作確認: data/topix500_sample.csv を指定\n"
            "  - 本格運用: JPX公式TOPIX500構成を data/topix500.csv に配置\n"
            "  推奨列: code,name[,effective_from,effective_to]"
        )

    # sizing のバリデーション
    sizing = cfg.get("sizing", {})
    adv_ratio = float(sizing.get("adv_ratio", 0.001))
    adv_ratio_cap = float(sizing.get("adv_ratio_cap", 0.002))
    if adv_ratio > adv_ratio_cap:
        raise ConfigError(
            f"sizing.adv_ratio ({adv_ratio}) が "
            f"sizing.adv_ratio_cap ({adv_ratio_cap}) を超えています"
        )

    # adv_window vs min_adv_periods バリデーション
    bt_min_adv = int(sizing.get("min_adv_periods", 1))
    bt_adv_win = int(sizing.get("adv_window", 20))
    if bt_min_adv > bt_adv_win:
        raise ConfigError(
            f"sizing.min_adv_periods ({bt_min_adv}) が "
            f"sizing.adv_window ({bt_adv_win}) を超えています"
        )

    bt_min_adv_bt = int(cfg.get("backtest", {}).get("min_adv_periods", 20))
    bt_adv_win_bt = int(cfg.get("backtest", {}).get("adv_window", 20))
    if bt_min_adv_bt > bt_adv_win_bt:
        raise ConfigError(
            f"backtest.min_adv_periods ({bt_min_adv_bt}) が "
            f"backtest.adv_window ({bt_adv_win_bt}) を超えています"
        )

    # notify.channel のバリデーション
    valid_channels = {"console", "discord"}
    channel = cfg.get("notify", {}).get("channel", "console")
    if channel not in valid_channels:
        raise ConfigError(f"notify.channel は {valid_channels} のいずれかである必要があります")

    if channel == "discord":
        webhook = cfg.get("notify", {}).get("discord_webhook", "")
        if not webhook:
            raise ConfigError(
                "discord チャンネル利用時は notify.discord_webhook か "
                "環境変数 DISCORD_WEBHOOK が必要です"
            )

    # shortability 運用ルール: 本番で売りを許可する設定は明示ログ
    if cfg["risk"].get("allow_short_without_confirmed_shortability", False):
        log.warning(
            "risk.allow_short_without_confirmed_shortability=true: "
            "shortability 未確認の売りを許可します（開発専用。本番禁止）。"
        )
    if cfg["backtest"].get("allow_unconfirmed_short_in_bt", False):
        log.warning(
            "backtest.allow_unconfirmed_short_in_bt=true: "
            "BT で shortability 未確認売りを許可します（開発専用）。"
        )

    if not cfg["backtest"].get("impact_k_is_calibrated", False):
        log.info("backtest.impact_k_is_calibrated=false: impact_k_bp は未較正です。")

    return cfg


def uses_approximate_turnover(cfg: dict) -> bool:
    """現行 data.source が近似 turnover かどうか。"""
    return str(cfg.get("data", {}).get("source", "")).lower() == "yfinance"


def guard_approximate_turnover(cfg: dict, *, context: str) -> None:
    """yfinance 近似 turnover を sizing/impact に使う処理を拒否する。

    allow_approximate_turnover=true のときのみ通過（明示オプトイン）。
    """
    if not uses_approximate_turnover(cfg):
        return
    if bool(cfg.get("data", {}).get("allow_approximate_turnover", False)):
        log.warning(
            "%s: yfinance 近似 turnover を明示許可して実行中 "
            "(allow_approximate_turnover=true)。本番利用は禁止。",
            context,
        )
        return
    raise ConfigError(
        f"{context}: data.source=yfinance の turnover は close*volume 近似です。"
        " sizing / market impact には使えません。\n"
        "  対応:\n"
        "    1) 本番: data.source=jquants と JQUANTS_API_KEY を設定\n"
        "    2) 疎通確認のみ: data.allow_approximate_turnover=true を明示"
        "（結果は信頼しない）"
    )


def enforce_short_policy_for_live(cfg: dict) -> None:
    """live で未確認売りを許可する設定を拒否（明示オプトイン以外）。

    開発で本当に必要な場合のみ
    risk.allow_short_without_confirmed_shortability=true を設定する。
    ここでは追加の hard fail はせず、設定値そのものを運用ルールの正とする。
    呼び出し側で risk_cfg に反映済みであることを前提に警告のみ行う。
    """
    if cfg.get("risk", {}).get("allow_short_without_confirmed_shortability", False):
        log.warning(
            "live short policy: 未確認売り許可中。 shortability.py 本実装前の本番運用は禁止。"
        )
