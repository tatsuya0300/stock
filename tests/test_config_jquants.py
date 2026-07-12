"""J-Quants config validation tests."""

from pathlib import Path

import pytest
import yaml

from jp_signal.config import ConfigError, load_config


def _write_config(tmp_path: Path, *, plan: str, sleep_sec: float | None) -> Path:
    """ヘルパー: テスト用設定ファイルを書き込む。"""
    universe_path = tmp_path / "universe.csv"
    universe_path.write_text("code,name\n7203,Toyota\n", encoding="utf-8")

    config = {
        "data": {
            "source": "jquants",
            "db_path": "./test.sqlite",
            "jquants_plan": plan,
            "jquants_sleep_sec": sleep_sec,
        },
        "universe": {
            "file": "./universe.csv",
        },
        "model": {
            "lookback": 5,
            "top_n": 1,
        },
        "backtest": {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "adv_window": 20,
            "min_adv_periods": 20,
        },
        "sizing": {
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "adv_window": 20,
            "min_adv_periods": 20,
        },
        "risk": {},
        "notify": {
            "channel": "console",
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_jquants_config_none_uses_plan_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """sleep_sec=None がプラン最小値になる。"""
    monkeypatch.setenv("JQUANTS_API_KEY", "dummy-api-key")
    path = _write_config(tmp_path, plan="free", sleep_sec=None)
    cfg = load_config(str(path))

    assert cfg["data"]["jquants_plan"] == "free"
    assert cfg["data"]["jquants_sleep_sec"] == pytest.approx(12.0)


def test_jquants_config_plan_is_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """プラン名が正規化される。"""
    monkeypatch.setenv("JQUANTS_API_KEY", "dummy-api-key")
    path = _write_config(tmp_path, plan=" LIGHT ", sleep_sec=None)
    cfg = load_config(str(path))

    assert cfg["data"]["jquants_plan"] == "light"
    assert cfg["data"]["jquants_sleep_sec"] == pytest.approx(1.0)


def test_jquants_config_rejects_unsafe_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """free plan で 1.0s は拒否される。"""
    monkeypatch.setenv("JQUANTS_API_KEY", "dummy-api-key")
    path = _write_config(tmp_path, plan="free", sleep_sec=1.0)

    with pytest.raises(ConfigError, match="rate-limit"):
        load_config(str(path))


def test_jquants_config_rejects_invalid_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """存在しないプラン名は拒否される。"""
    monkeypatch.setenv("JQUANTS_API_KEY", "dummy-api-key")
    path = _write_config(tmp_path, plan="enterprise", sleep_sec=None)

    with pytest.raises(ConfigError, match="rate-limit"):
        load_config(str(path))
