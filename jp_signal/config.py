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
from datetime import date
from pathlib import Path
from typing import Any

from .jquants_limits import resolve_jquants_sleep_sec

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

    # J-Quants 設定
    "jquants_plan": "free",
    # None なら契約プラン別の最小安全間隔を使用する
    "jquants_sleep_sec": None,

    # datasource 段階でデータ品質を厳格に検証する
    "strict_data_quality": True,

    # yfinance の一括取得銘柄数
    "yfinance_chunk_size": 50,
}

_DEFAULT_BACKTEST = {
    "impact_k_is_calibrated": False,
    "allow_unconfirmed_short_in_bt": False,
    "min_adv_periods": 20,
}


class ConfigError(ValueError):
    """設定不備。"""


def _resolve_path(
    raw_path: str | None,
    *,
    config_path: Path,
) -> str | None:
    """相対パスをconfigファイルの配置場所基準で絶対パス化する。"""
    if raw_path is None:
        return None

    value = str(raw_path).strip()
    if not value:
        return value

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = config_path.parent / path

    return str(path.resolve())


def _as_float(
    mapping: dict[str, Any],
    key: str,
    *,
    section: str,
    default: float | None = None,
) -> float:
    raw = mapping.get(key, default)

    if raw is None:
        raise ConfigError(f"{section}.{key} が設定されていません")

    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{section}.{key} は数値である必要があります: {raw!r}") from exc


def _as_int(
    mapping: dict[str, Any],
    key: str,
    *,
    section: str,
    default: int | None = None,
) -> int:
    raw = mapping.get(key, default)

    if raw is None:
        raise ConfigError(f"{section}.{key} が設定されていません")

    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{section}.{key} は整数である必要があります: {raw!r}") from exc


def _validate_non_negative(
    mapping: dict[str, Any],
    keys: list[str],
    *,
    section: str,
) -> None:
    for key in keys:
        if key not in mapping:
            continue

        value = _as_float(
            mapping,
            key,
            section=section,
        )

        if value < 0:
            raise ConfigError(f"{section}.{key} は負の値にできません: {value}")


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

    # 相対パスをconfigファイルの配置場所基準で解決する
    cfg["data"]["db_path"] = _resolve_path(
        cfg["data"].get("db_path"),
        config_path=p,
    )

    if "file" in cfg.get("universe", {}):
        cfg["universe"]["file"] = _resolve_path(
            cfg["universe"]["file"],
            config_path=p,
        )

    if "output_dir" in cfg.get("backtest", {}):
        cfg["backtest"]["output_dir"] = _resolve_path(
            cfg["backtest"]["output_dir"],
            config_path=p,
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

    _validate_config_values(cfg)

    return cfg


def _validate_config_values(cfg: dict) -> None:
    """実行前に設定値の意味的整合性を検証する。"""
    data = cfg["data"]
    backtest = cfg["backtest"]
    sizing = cfg["sizing"]
    risk = cfg.get("risk", {})
    model = cfg.get("model", {})
    data_quality = cfg.get("data_quality", {})

    source = str(data.get("source", "")).strip().lower()

    if source == "jquants":
        raw_sleep_sec = data.get("jquants_sleep_sec")
        sleep_sec = None if raw_sleep_sec is None else _as_float(data, "jquants_sleep_sec", section="data")

        try:
            normalized_plan, resolved_sleep_sec = resolve_jquants_sleep_sec(
                plan=str(data.get("jquants_plan", "free")),
                sleep_sec=sleep_sec,
            )
        except ValueError as exc:
            raise ConfigError(f"J-Quants rate-limit設定が不正です: {exc}") from exc

        # 後続処理で同じ値を利用できるよう正規化する
        data["jquants_plan"] = normalized_plan
        data["jquants_sleep_sec"] = resolved_sleep_sec

    yfinance_chunk_size = _as_int(data, "yfinance_chunk_size", section="data", default=50)
    if yfinance_chunk_size < 1:
        raise ConfigError(f"data.yfinance_chunk_size は1以上である必要があります: {yfinance_chunk_size}")

    if "start" in backtest and "end" in backtest:
        try:
            start = date.fromisoformat(str(backtest["start"]))
            end = date.fromisoformat(str(backtest["end"]))
        except ValueError as exc:
            raise ConfigError("backtest.start/end はYYYY-MM-DD形式である必要があります") from exc

        if start > end:
            raise ConfigError(f"backtest.startはend以前である必要があります: {start} > {end}")

    _validate_non_negative(
        backtest,
        [
            "impact_k_bp",
            "annual_interest_rate",
            "short_lending_rate",
            "commission_bp",
            "half_spread_bp",
            "risk_free_rate",
        ],
        section="backtest",
    )

    initial_capital = _as_float(
        backtest,
        "initial_capital",
        section="backtest",
        default=100_000_000,
    )
    if initial_capital <= 0:
        raise ConfigError(
            f"backtest.initial_capital は正の値である必要があります: {initial_capital}"
        )

    _validate_non_negative(
        sizing,
        ["adv_ratio", "adv_ratio_cap"],
        section="sizing",
    )

    adv_ratio = _as_float(
        sizing,
        "adv_ratio",
        section="sizing",
        default=0.001,
    )
    adv_ratio_cap = _as_float(
        sizing,
        "adv_ratio_cap",
        section="sizing",
        default=0.002,
    )
    if adv_ratio > adv_ratio_cap:
        raise ConfigError(
            f"sizing.adv_ratio ({adv_ratio}) が "
            f"sizing.adv_ratio_cap ({adv_ratio_cap}) を超えています"
        )

    reference_price_buffer_ratio = _as_float(
        sizing,
        "reference_price_buffer_ratio",
        section="sizing",
        default=0.0,
    )
    if not 0 <= reference_price_buffer_ratio <= 1.0:
        raise ConfigError(
            f"sizing.reference_price_buffer_ratio "
            f"({reference_price_buffer_ratio}) は0〜1の範囲である必要があります"
        )

    sizing_adv_window = _as_int(
        sizing,
        "adv_window",
        section="sizing",
        default=20,
    )
    sizing_min_periods = _as_int(
        sizing,
        "min_adv_periods",
        section="sizing",
        default=1,
    )

    if sizing_adv_window < sizing_min_periods:
        raise ConfigError(
            f"sizing.adv_window ({sizing_adv_window}) が "
            f"sizing.min_adv_periods ({sizing_min_periods}) 未満です"
        )

    # backtest 専用の adv_window/min_adv_periods
    bt_adv_window = _as_int(
        backtest,
        "adv_window",
        section="backtest",
        default=sizing_adv_window,
    )
    bt_min_periods = _as_int(
        backtest,
        "min_adv_periods",
        section="backtest",
        default=sizing_min_periods,
    )

    if bt_adv_window < bt_min_periods:
        raise ConfigError(
            f"backtest.adv_window ({bt_adv_window}) が "
            f"backtest.min_adv_periods ({bt_min_periods}) 未満です"
        )

    max_single_name = _as_float(
        risk,
        "max_single_name_exposure_yen",
        section="risk",
        default=20_000_000,
    )
    max_gross = _as_float(
        risk,
        "max_gross_exposure_yen",
        section="risk",
        default=100_000_000,
    )

    if max_single_name > max_gross:
        raise ConfigError(
            "risk.max_single_name_exposure_yen はmax_gross_exposure_yen以下である必要があります"
        )

    for key in [
        "price_coverage_min",
        "lookback_coverage_min",
        "turnover_coverage_min",
    ]:
        if key not in data_quality:
            continue
        value = _as_float(
            data_quality,
            key,
            section="data_quality",
        )
        if not 0 < value <= 1.0:
            raise ConfigError(f"data_quality.{key} は0〜1の範囲である必要があります: {value}")

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

    # shortability 運用ルール
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

    # モデル validation（存在する場合のみ）
    if "lookback" in model:
        lookback = _as_int(
            model,
            "lookback",
            section="model",
        )
        if lookback < 1:
            raise ConfigError(f"model.lookback は1以上である必要があります: {lookback}")


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
