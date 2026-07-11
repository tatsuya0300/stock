"""Tests for config path resolution and value validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jp_signal.config import ConfigError, load_config


def _base_config(tmp_path: Path) -> dict:
    universe = tmp_path / "universe.csv"
    universe.write_text("code,name\n7203,Toyota\n", encoding="utf-8")

    return {
        "data": {
            "source": "yfinance",
            "db_path": "./db/test.sqlite",
            "allow_approximate_turnover": True,
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
            "initial_capital": 100_000_000,
            "holding_days": 1,
            "adv_window": 20,
            "min_adv_periods": 20,
            "impact_k_bp": 30,
            "commission_bp": 1,
            "half_spread_bp": 5,
            "output_dir": "./output",
        },
        "sizing": {
            "adv_ratio": 0.001,
            "adv_ratio_cap": 0.002,
            "adv_window": 20,
            "min_adv_periods": 20,
        },
        "risk": {
            "max_orders_per_day": 10,
            "max_gross_exposure_yen": 100_000_000,
            "max_single_name_exposure_yen": 20_000_000,
        },
        "notify": {
            "channel": "console",
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def test_relative_paths_are_resolved_from_config_directory(tmp_path):
    config = _base_config(tmp_path)
    path = _write_config(tmp_path, config)

    loaded = load_config(str(path))

    assert loaded["universe"]["file"] == str(
        (tmp_path / "universe.csv").resolve()
    )
    assert loaded["data"]["db_path"] == str(
        (tmp_path / "db/test.sqlite").resolve()
    )
    assert loaded["backtest"]["output_dir"] == str(
        (tmp_path / "output").resolve()
    )


def test_backtest_start_must_not_be_after_end(tmp_path):
    config = _base_config(tmp_path)
    config["backtest"]["start"] = "2025-01-01"
    config["backtest"]["end"] = "2024-01-01"

    with pytest.raises(ConfigError, match="start"):
        load_config(str(_write_config(tmp_path, config)))


def test_negative_cost_is_rejected(tmp_path):
    config = _base_config(tmp_path)
    config["backtest"]["commission_bp"] = -1

    with pytest.raises(ConfigError, match="commission_bp"):
        load_config(str(_write_config(tmp_path, config)))


def test_invalid_coverage_threshold_is_rejected(tmp_path):
    config = _base_config(tmp_path)
    config["data_quality"] = {
        "price_coverage_min": 1.1,
    }

    with pytest.raises(ConfigError, match="price_coverage_min"):
        load_config(str(_write_config(tmp_path, config)))
