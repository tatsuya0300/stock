"""研究用バックテストの事前検証。

条件を満たさないバックテストをresearch-validとして扱わない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class ResearchPreflightResult:
    issues: list[PreflightIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "ERROR" for issue in self.issues)

    def add_error(self, code: str, message: str) -> None:
        self.issues.append(
            PreflightIssue(code=code, severity="ERROR", message=message)
        )

    def add_warning(self, code: str, message: str) -> None:
        self.issues.append(
            PreflightIssue(code=code, severity="WARNING", message=message)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def raise_if_failed(self) -> None:
        if self.passed:
            return

        errors = [
            f"{issue.code}: {issue.message}"
            for issue in self.issues
            if issue.severity == "ERROR"
        ]

        raise RuntimeError(
            "Research preflight failed:\n- " + "\n- ".join(errors)
        )


def validate_universe_pit(
    universe: pd.DataFrame,
    result: ResearchPreflightResult,
) -> None:
    required = {"code", "effective_from", "effective_to", "available_at"}

    missing = required - set(universe.columns)

    if missing:
        result.add_error(
            "UNIVERSE_NOT_PIT",
            "universeにPIT列が不足しています: " f"{sorted(missing)}",
        )
        return

    frame = universe.copy()

    for column in ["effective_from", "effective_to", "available_at"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)

    if frame[["effective_from", "available_at"]].isna().any().any():
        result.add_error(
            "UNIVERSE_INVALID_TIMESTAMP",
            "effective_from/available_atに無効なtimestampがあります",
        )

    invalid_range = frame["effective_to"].notna() & (
        frame["effective_to"] < frame["effective_from"]
    )

    if invalid_range.any():
        result.add_error(
            "UNIVERSE_INVALID_RANGE",
            "effective_toがeffective_fromより前の行があります",
        )


def validate_prices_pit(
    prices: pd.DataFrame | None,
    result: ResearchPreflightResult,
) -> None:
    if prices is None:
        result.add_error(
            "PRICES_NOT_PIT", "価格データがNoneです",
        )
        return

    required = {"code", "date", "available_at"}

    missing = required - set(prices.columns)

    if missing:
        result.add_error(
            "PRICES_NOT_PIT",
            "価格データにPIT列が不足しています: " f"{sorted(missing)}",
        )
        return

    frame = prices.copy()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], errors="coerce", utc=True
    )

    if frame[["date", "available_at"]].isna().any().any():
        result.add_error(
            "PRICE_INVALID_TIMESTAMP",
            "価格データに無効なdate/available_atがあります",
        )


def validate_trade_outputs(
    *,
    trades: pd.DataFrame,
    open_positions: pd.DataFrame,
    rejected_orders: pd.DataFrame,
    result: ResearchPreflightResult,
) -> None:
    if open_positions is not None and not open_positions.empty:
        result.add_error(
            "UNRESOLVED_POSITIONS",
            f"バックテスト終了時に{len(open_positions)}件の未決済ポジションがあります",
        )

    if trades is None or trades.empty:
        result.add_error("NO_TRADES", "取引が0件です")
        return

    if "forced_exit_reason" in trades.columns:
        forced = trades[
            trades["forced_exit_reason"].notna()
            & trades["forced_exit_reason"].astype(str).ne("")
        ]

        if not forced.empty:
            result.add_error(
                "FORCED_EXIT_PRESENT",
                f"{len(forced)}件の強制決済があります。主解析では採用できません",
            )

    if rejected_orders is not None and not rejected_orders.empty:
        system_reasons = {
            "INVALID_EQUITY",
            "INVALID_EQUITY_AT_OPEN",
            "NO_EXIT_PRICE_DEFERRED",
            "NO_EXIT_LIQUIDITY_DEFERRED",
        }

        reason = rejected_orders.get("reason", pd.Series(dtype=str)).astype(str)

        affected = reason[reason.str.contains("|".join(system_reasons), regex=True)]

        if not affected.empty:
            result.add_error(
                "SYSTEM_REJECTION_PRESENT",
                "価格・流動性・equity不正によるsystem rejectionがあります",
            )


def validate_research_config(
    cfg: dict,
    result: ResearchPreflightResult,
) -> None:
    data_cfg = cfg.get("data", {})
    bt_cfg = cfg.get("backtest", {})
    research_cfg = cfg.get("research", {})

    if data_cfg.get("price_vintage_mode") != "point_in_time":
        result.add_error(
            "LATEST_SNAPSHOT_FORBIDDEN",
            "research-valid BTではdata.price_vintage_mode=point_in_timeが必要です",
        )

    if data_cfg.get("source") != "jquants":
        result.add_error(
            "NON_PRODUCTION_DATASOURCE",
            "research-valid BTではJ-Quants等の正式な売買代金データが必要です",
        )

    if not bt_cfg.get("impact_k_is_calibrated", False):
        result.add_error(
            "IMPACT_NOT_CALIBRATED",
            "impact係数が実fillsで較正されていません",
        )

    if not bt_cfg.get("require_corporate_actions", False):
        result.add_error(
            "CORPORATE_ACTIONS_OPTIONAL",
            "corporate actionを必須にしてください",
        )

    if not research_cfg.get("trial_registry_enabled", False):
        result.add_error(
            "TRIAL_REGISTRY_DISABLED",
            "全試行を保存するtrial registryが必要です",
        )

    split = research_cfg.get("split", {})

    required_split = {
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    }

    missing = required_split - set(split.keys())

    if missing:
        result.add_error(
            "RESEARCH_SPLIT_MISSING",
            "train/validation/test期間が不足しています: " f"{sorted(missing)}",
        )

    if research_cfg.get("allow_test_reuse", False):
        result.add_error(
            "TEST_REUSE_ENABLED",
            "untouched testの再利用を許可できません",
        )


def run_research_preflight(
    *,
    cfg: dict,
    universe: pd.DataFrame,
    prices: pd.DataFrame | None,
    trades: pd.DataFrame,
    open_positions: pd.DataFrame,
    rejected_orders: pd.DataFrame,
) -> ResearchPreflightResult:
    result = ResearchPreflightResult()

    validate_research_config(cfg, result)
    validate_universe_pit(universe, result)
    validate_prices_pit(prices, result)
    validate_trade_outputs(
        trades=trades,
        open_positions=open_positions,
        rejected_orders=rejected_orders,
        result=result,
    )

    return result
