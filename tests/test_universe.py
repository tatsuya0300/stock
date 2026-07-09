"""universe loader tests."""

import pandas as pd
import pytest

from jp_signal.universe import load_universe, normalize_code


def test_normalize_code():
    assert normalize_code("7203") == "7203"
    assert normalize_code("7203.T") == "7203"
    assert normalize_code("123") == "0123"


def test_load_static_universe(tmp_path):
    p = tmp_path / "univ.csv"
    pd.DataFrame(
        {
            "code": ["7203", "6758.T"],
            "name": ["Toyota", "Sony"],
        }
    ).to_csv(p, index=False)

    df = load_universe(str(p))

    assert df["code"].tolist() == ["6758", "7203"]


def test_duplicate_code_rejected(tmp_path):
    p = tmp_path / "univ.csv"
    pd.DataFrame(
        {
            "code": ["7203", "7203.T"],
            "name": ["A", "B"],
        }
    ).to_csv(p, index=False)

    with pytest.raises(ValueError):
        load_universe(str(p))


def test_point_in_time_universe(tmp_path):
    p = tmp_path / "univ.csv"
    pd.DataFrame(
        {
            "code": ["1111", "2222"],
            "name": ["A", "B"],
            "effective_from": ["2020-01-01", "2025-01-01"],
            "effective_to": ["2024-12-31", ""],
        }
    ).to_csv(p, index=False)

    df_2024 = load_universe(str(p), as_of="2024-06-01")
    df_2025 = load_universe(str(p), as_of="2025-06-01")

    assert df_2024["code"].tolist() == ["1111"]
    assert df_2025["code"].tolist() == ["2222"]


def test_load_universe_accepts_cfg_dict(tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("code,name\n7203,トヨタ\n", encoding="utf-8")
    from jp_signal.universe import load_universe

    df = load_universe({"file": str(p)})
    assert list(df["code"]) == ["7203"]
