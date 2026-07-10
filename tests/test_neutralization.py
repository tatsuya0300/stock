from __future__ import annotations

import pandas as pd
import pytest

from jp_signal.neutralization import neutralize_scores


def test_sector_neutral_scores_have_reduced_sector_bias():
    signals = pd.DataFrame(
        [
            {"code": "A", "score": 3.0},
            {"code": "B", "score": 1.0},
            {"code": "C", "score": 4.0},
            {"code": "D", "score": 2.0},
        ]
    )

    exposures = pd.DataFrame(
        [
            {"code": "A", "sector": "X"},
            {"code": "B", "sector": "X"},
            {"code": "C", "sector": "Y"},
            {"code": "D", "sector": "Y"},
        ]
    )

    result = neutralize_scores(
        signals,
        exposures,
        ridge=1.0,
    )

    sector_means = result.groupby("sector")["neutral_score"].mean()

    # With ridge regularization, sector means are closer to zero than raw scores.
    raw_sector_means = result.groupby("sector")["score"].mean()
    raw_spread = abs(raw_sector_means["X"] - raw_sector_means["Y"])
    neutral_spread = abs(sector_means["X"] - sector_means["Y"])

    assert neutral_spread < raw_spread
    assert "neutral_score" in result.columns


def test_neutralize_without_categorical():
    signals = pd.DataFrame(
        [
            {"code": "A", "score": 1.0},
            {"code": "B", "score": 2.0},
        ]
    )

    exposures = pd.DataFrame(
        [
            {"code": "A"},
            {"code": "B"},
        ]
    )

    result = neutralize_scores(
        signals,
        exposures,
        categorical_cols=(),
        numeric_cols=(),
    )

    assert "neutral_score" in result.columns
    assert result["neutral_score"].sum() == pytest.approx(0.0, abs=1e-10)
