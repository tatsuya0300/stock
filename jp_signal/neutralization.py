"""Sector / risk-factor score neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_scores(
    signals: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    code_col: str = "code",
    score_col: str = "score",
    categorical_cols: tuple[str, ...] = ("sector",),
    numeric_cols: tuple[str, ...] = (),
    ridge: float = 1.0,
) -> pd.DataFrame:
    """scoreをsector/beta等のexposureに対して残差化する。

    categorical exposureはone-hot化する。
    戻り値のscoreはneutral_score列に保存する。
    """
    if ridge < 0:
        raise ValueError(f"ridge must be >= 0: {ridge}")

    required_signal = {
        code_col,
        score_col,
    }
    missing_signal = required_signal - set(signals.columns)

    if missing_signal:
        raise ValueError(f"signals missing columns: {sorted(missing_signal)}")

    required_exposure = {
        code_col,
        *categorical_cols,
        *numeric_cols,
    }
    missing_exposure = required_exposure - set(exposures.columns)

    if missing_exposure:
        raise ValueError(f"exposures missing columns: {sorted(missing_exposure)}")

    merged = signals.merge(
        exposures[
            [
                code_col,
                *categorical_cols,
                *numeric_cols,
            ]
        ],
        on=code_col,
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        return merged.assign(neutral_score=pd.Series(dtype=float))

    y = pd.to_numeric(
        merged[score_col],
        errors="coerce",
    )

    valid = y.notna()

    for column in numeric_cols:
        valid &= pd.to_numeric(
            merged[column],
            errors="coerce",
        ).notna()

    merged = merged.loc[valid].copy()
    y = y.loc[valid].to_numpy(dtype=float)

    design_parts: list[pd.DataFrame] = []

    for column in categorical_cols:
        dummies = pd.get_dummies(
            merged[column].astype("string"),
            prefix=column,
            dtype=float,
        )
        design_parts.append(dummies)

    if numeric_cols:
        numeric = merged[list(numeric_cols)].apply(
            pd.to_numeric,
            errors="raise",
        )
        design_parts.append(numeric.astype(float))

    if not design_parts:
        merged["neutral_score"] = y - float(np.mean(y))
        return merged

    design = pd.concat(
        design_parts,
        axis=1,
    ).to_numpy(dtype=float)

    # interceptを明示的に追加する。
    design = np.column_stack(
        [
            np.ones(len(design)),
            design,
        ]
    )

    identity = np.eye(
        design.shape[1],
        dtype=float,
    )
    identity[0, 0] = 0.0  # interceptはpenalizeしない

    coefficients = np.linalg.solve(
        design.T @ design + ridge * identity,
        design.T @ y,
    )

    fitted = design @ coefficients
    residual = y - fitted

    # 日次cross-section内で平均0に補正。
    residual -= residual.mean()

    merged["neutral_score"] = residual

    return merged
