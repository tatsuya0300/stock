"""config tests."""

import pytest
import yaml

from jp_signal.config import (
    ConfigError,
    guard_approximate_turnover,
    load_config,
    uses_approximate_turnover,
)


def _base_config(universe_file: str):
    return {
        "data": {
            "source": "yfinance",
            "db_path": "./data/test.sqlite",
            "allow_approximate_turnover": False,
        },
        "universe": {
            "file": universe_file,
        },
        "backtest": {
            "start": "2022-01-01",
            "end": "2024-12-31",
            "impact_k_bp": 30.0,
            "annual_interest_rate": 0.02,
            "short_lending_rate": 0.02,
        },
        "sizing": {
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "unit": 100,
            "market_open_unit_cap": 50,
        },
        "notify": {
            "channel": "console",
        },
        "risk": {
            "allow_short_without_confirmed_shortability": False,
        },
    }


def _write_universe(tmp_path):
    univ = tmp_path / "univ.csv"
    univ.write_text("code,name\n7203,Toyota\n", encoding="utf-8")
    return str(univ)


def test_load_config_valid(tmp_path):
    univ = _write_universe(tmp_path)
    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_base_config(univ), f)

    cfg = load_config(str(p))
    assert cfg["data"]["source"] == "yfinance"
    assert cfg["data"]["allow_approximate_turnover"] is False
    assert cfg["risk"]["allow_short_without_confirmed_shortability"] is False


def test_invalid_data_source_rejected(tmp_path):
    univ = _write_universe(tmp_path)
    cfg = _base_config(univ)
    cfg["data"]["source"] = "bad"

    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises((ValueError, ConfigError)):
        load_config(str(p))


def test_adv_ratio_over_cap_rejected(tmp_path):
    univ = _write_universe(tmp_path)
    cfg = _base_config(univ)
    cfg["sizing"]["adv_ratio"] = 0.003
    cfg["sizing"]["adv_ratio_cap"] = 0.002

    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises((ValueError, ConfigError)):
        load_config(str(p))


def test_discord_requires_webhook(tmp_path):
    univ = _write_universe(tmp_path)
    cfg = _base_config(univ)
    cfg["notify"]["channel"] = "discord"
    cfg["notify"].pop("discord_webhook", None)

    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises((ValueError, ConfigError)):
        load_config(str(p))


def test_missing_universe_file_rejected(tmp_path):
    cfg = _base_config(str(tmp_path / "missing.csv"))
    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises((ValueError, ConfigError)):
        load_config(str(p))


def test_guard_approximate_turnover_hard_fail(tmp_path):
    univ = _write_universe(tmp_path)
    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_base_config(univ), f)
    cfg = load_config(str(p))

    assert uses_approximate_turnover(cfg) is True
    with pytest.raises(ConfigError):
        guard_approximate_turnover(cfg, context="test")


def test_guard_approximate_turnover_opt_in(tmp_path):
    univ = _write_universe(tmp_path)
    raw = _base_config(univ)
    raw["data"]["allow_approximate_turnover"] = True
    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f)
    cfg = load_config(str(p))

    # 例外にならないこと
    guard_approximate_turnover(cfg, context="test")
