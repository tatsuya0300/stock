"""config tests."""

import pytest
import yaml

from jp_signal.config import load_config


def _base_config():
    return {
        "data": {
            "source": "yfinance",
            "db_path": "./data/test.sqlite",
        },
        "universe": {
            "file": "./data/topix500.csv",
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
    }


def test_load_config_valid(tmp_path):
    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_base_config(), f)

    cfg = load_config(str(p))

    assert cfg["data"]["source"] == "yfinance"


def test_invalid_data_source_rejected(tmp_path):
    cfg = _base_config()
    cfg["data"]["source"] = "bad"

    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(ValueError):
        load_config(str(p))


def test_adv_ratio_over_cap_rejected(tmp_path):
    cfg = _base_config()
    cfg["sizing"]["adv_ratio"] = 0.003
    cfg["sizing"]["adv_ratio_cap"] = 0.002

    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(ValueError):
        load_config(str(p))


def test_discord_requires_webhook(tmp_path):
    cfg = _base_config()
    cfg["notify"]["channel"] = "discord"
    cfg["notify"]["discord_webhook"] = ""

    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(ValueError):
        load_config(str(p))
