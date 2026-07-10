"""Market-impact calibration from realized fills."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ImpactCalibrationResult:
    observations: int
    intercept_bp: float
    impact_k_bp: float
    r_squared: float
    rmse_bp: float
    participation_min: float
    participation_median: float
    participation_max: float


def calibrate_square_root_impact(
    fills: pd.DataFrame,
    *,
    expected_price_col: str = "expected_price",
    fill_price_col: str = "price",
    qty_col: str = "qty",
    adv_col: str = "adv",
    side_col: str = "side",
    minimum_observations: int = 30,
) -> ImpactCalibrationResult:
    """実約定から slippage = intercept + k*sqrt(participation) を推定する。

    BUY:
        (fill - expected) / expected

    SELL:
        (expected - fill) / expected

    正の値ほど不利なslippage。
    """
    required = {
        expected_price_col,
        fill_price_col,
        qty_col,
        adv_col,
        side_col,
    }
    missing = required - set(fills.columns)

    if missing:
        raise ValueError(f"fills missing columns: {sorted(missing)}")

    frame = fills.copy()

    expected = pd.to_numeric(
        frame[expected_price_col],
        errors="coerce",
    )
    fill_price = pd.to_numeric(
        frame[fill_price_col],
        errors="coerce",
    )
    qty = pd.to_numeric(
        frame[qty_col],
        errors="coerce",
    )
    adv = pd.to_numeric(
        frame[adv_col],
        errors="coerce",
    )
    side = frame[side_col].astype(str).str.upper()

    valid = (
        expected.notna()
        & fill_price.notna()
        & qty.notna()
        & adv.notna()
        & (expected > 0)
        & (fill_price > 0)
        & (qty > 0)
        & (adv > 0)
        & side.isin({"BUY", "SELL"})
    )

    expected = expected[valid].to_numpy(dtype=float)
    fill_price = fill_price[valid].to_numpy(dtype=float)
    qty = qty[valid].to_numpy(dtype=float)
    adv = adv[valid].to_numpy(dtype=float)
    side = side[valid]

    if len(expected) < minimum_observations:
        raise ValueError(f"need at least {minimum_observations} valid fills, got {len(expected)}")

    raw_slippage = np.where(
        side == "BUY",
        (fill_price - expected) / expected,
        (expected - fill_price) / expected,
    )
    slippage_bp = raw_slippage * 10_000.0

    notional = expected * qty
    participation = notional / adv
    sqrt_participation = np.sqrt(participation)

    design = np.column_stack(
        [
            np.ones(len(sqrt_participation)),
            sqrt_participation,
        ]
    )

    coefficients, *_rest = np.linalg.lstsq(
        design,
        slippage_bp,
        rcond=None,
    )

    fitted = design @ coefficients
    residual = slippage_bp - fitted

    ss_total = float(np.sum(np.square(slippage_bp - slippage_bp.mean())))
    ss_residual = float(np.sum(np.square(residual)))

    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0

    return ImpactCalibrationResult(
        observations=len(expected),
        intercept_bp=float(coefficients[0]),
        impact_k_bp=float(coefficients[1]),
        r_squared=float(r_squared),
        rmse_bp=float(np.sqrt(np.mean(np.square(residual)))),
        participation_min=float(participation.min()),
        participation_median=float(np.median(participation)),
        participation_max=float(participation.max()),
    )
