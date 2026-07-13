"""Research preflight gate tests."""

import pandas as pd
import pytest

from jp_signal.research.preflight import (
    ResearchPreflightResult,
    PreflightIssue,
    validate_research_config,
    validate_universe_pit,
    validate_prices_pit,
    validate_trade_outputs,
    run_research_preflight,
)


class TestResearchPreflightResult:
    def test_passed_when_no_errors(self) -> None:
        result = ResearchPreflightResult()
        assert result.passed

    def test_failed_when_error_exists(self) -> None:
        result = ResearchPreflightResult()
        result.add_error("TEST", "test error")
        assert not result.passed

    def test_warning_does_not_fail(self) -> None:
        result = ResearchPreflightResult()
        result.add_warning("TEST", "test warning")
        assert result.passed

    def test_raise_if_failed_raises(self) -> None:
        result = ResearchPreflightResult()
        result.add_error("ERR1", "error 1")
        result.add_error("ERR2", "error 2")
        with pytest.raises(RuntimeError, match="Research preflight failed"):
            result.raise_if_failed()

    def test_raise_if_failed_ok(self) -> None:
        result = ResearchPreflightResult()
        result.raise_if_failed()  # should not raise

    def test_to_dict(self) -> None:
        result = ResearchPreflightResult()
        result.add_error("ERR", "problem")
        d = result.to_dict()
        assert d["passed"] is False
        assert len(d["issues"]) == 1
        assert d["issues"][0]["code"] == "ERR"

    def test_issue_to_dict(self) -> None:
        issue = PreflightIssue(code="C1", severity="ERROR", message="msg")
        d = issue.to_dict()
        assert d == {"code": "C1", "severity": "ERROR", "message": "msg"}


class TestValidateResearchConfig:
    def test_pass_with_valid_config(self) -> None:
        cfg = {
            "data": {
                "price_vintage_mode": "point_in_time",
                "source": "jquants",
            },
            "backtest": {
                "impact_k_is_calibrated": True,
                "require_corporate_actions": True,
            },
            "research": {
                "trial_registry_enabled": True,
                "allow_test_reuse": False,
                "split": {
                    "train_start": "2010-01-01",
                    "train_end": "2018-12-31",
                    "validation_start": "2019-01-01",
                    "validation_end": "2022-12-31",
                    "test_start": "2023-01-01",
                    "test_end": "2025-12-31",
                },
            },
        }
        result = ResearchPreflightResult()
        validate_research_config(cfg, result)
        assert result.passed, f"unexpected issues: {result.issues}"

    def test_fail_latest_snapshot(self) -> None:
        cfg = {"data": {"price_vintage_mode": "latest_snapshot", "source": "jquants"}, "backtest": {}, "research": {}}
        result = ResearchPreflightResult()
        validate_research_config(cfg, result)
        assert any(i.code == "LATEST_SNAPSHOT_FORBIDDEN" for i in result.issues)

    def test_fail_non_jquants(self) -> None:
        cfg = {"data": {"price_vintage_mode": "point_in_time", "source": "yfinance"}, "backtest": {}, "research": {}}
        result = ResearchPreflightResult()
        validate_research_config(cfg, result)
        assert any(i.code == "NON_PRODUCTION_DATASOURCE" for i in result.issues)

    def test_fail_no_calibrated_impact(self) -> None:
        cfg = {"data": {"price_vintage_mode": "point_in_time", "source": "jquants"}, "backtest": {}, "research": {}}
        result = ResearchPreflightResult()
        validate_research_config(cfg, result)
        assert any(i.code == "IMPACT_NOT_CALIBRATED" for i in result.issues)

    def test_fail_no_corporate_actions(self) -> None:
        cfg = {
            "data": {"price_vintage_mode": "point_in_time", "source": "jquants"},
            "backtest": {"impact_k_is_calibrated": True},
            "research": {},
        }
        result = ResearchPreflightResult()
        validate_research_config(cfg, result)
        assert any(i.code == "CORPORATE_ACTIONS_OPTIONAL" for i in result.issues)

    def test_fail_no_trial_registry(self) -> None:
        cfg = {
            "data": {"price_vintage_mode": "point_in_time", "source": "jquants"},
            "backtest": {"impact_k_is_calibrated": True, "require_corporate_actions": True},
            "research": {"trial_registry_enabled": False, "allow_test_reuse": False, "split": {}},
        }
        result = ResearchPreflightResult()
        validate_research_config(cfg, result)
        assert any(i.code == "TRIAL_REGISTRY_DISABLED" for i in result.issues)

    def test_fail_missing_split(self) -> None:
        cfg = {
            "data": {"price_vintage_mode": "point_in_time", "source": "jquants"},
            "backtest": {"impact_k_is_calibrated": True, "require_corporate_actions": True},
            "research": {"trial_registry_enabled": True, "allow_test_reuse": False, "split": {}},
        }
        result = ResearchPreflightResult()
        validate_research_config(cfg, result)
        assert any(i.code == "RESEARCH_SPLIT_MISSING" for i in result.issues)

    def test_fail_test_reuse(self) -> None:
        cfg = {
            "data": {"price_vintage_mode": "point_in_time", "source": "jquants"},
            "backtest": {"impact_k_is_calibrated": True, "require_corporate_actions": True},
            "research": {
                "trial_registry_enabled": True,
                "allow_test_reuse": True,
                "split": {
                    "train_start": "2010-01-01",
                    "train_end": "2018-12-31",
                    "validation_start": "2019-01-01",
                    "validation_end": "2022-12-31",
                    "test_start": "2023-01-01",
                    "test_end": "2025-12-31",
                },
            },
        }
        result = ResearchPreflightResult()
        validate_research_config(cfg, result)
        assert any(i.code == "TEST_REUSE_ENABLED" for i in result.issues)


class TestValidateUniversePit:
    def test_pass(self) -> None:
        universe = pd.DataFrame({
            "code": ["1111"],
            "effective_from": ["2020-01-01"],
            "effective_to": [pd.NA],
            "available_at": ["2020-01-01 08:00:00+00:00"],
        })
        result = ResearchPreflightResult()
        validate_universe_pit(universe, result)
        assert result.passed

    def test_missing_columns(self) -> None:
        universe = pd.DataFrame({"code": ["1111"]})
        result = ResearchPreflightResult()
        validate_universe_pit(universe, result)
        assert any(i.code == "UNIVERSE_NOT_PIT" for i in result.issues)

    def test_invalid_timestamp(self) -> None:
        universe = pd.DataFrame({
            "code": ["1111"],
            "effective_from": ["not-a-date"],
            "effective_to": [pd.NA],
            "available_at": ["2020-01-01"],
        })
        result = ResearchPreflightResult()
        validate_universe_pit(universe, result)
        assert any(i.code == "UNIVERSE_INVALID_TIMESTAMP" for i in result.issues)

    def test_invalid_range(self) -> None:
        universe = pd.DataFrame({
            "code": ["1111"],
            "effective_from": ["2020-06-01"],
            "effective_to": ["2020-01-01"],
            "available_at": ["2020-01-01"],
        })
        result = ResearchPreflightResult()
        validate_universe_pit(universe, result)
        assert any(i.code == "UNIVERSE_INVALID_RANGE" for i in result.issues)


class TestValidatePricesPit:
    def test_pass(self) -> None:
        prices = pd.DataFrame({
            "code": ["1111"],
            "date": ["2020-01-10"],
            "available_at": ["2020-01-10 08:00:00+00:00"],
        })
        result = ResearchPreflightResult()
        validate_prices_pit(prices, result)
        assert result.passed

    def test_none(self) -> None:
        result = ResearchPreflightResult()
        validate_prices_pit(None, result)
        assert any(i.code == "PRICES_NOT_PIT" for i in result.issues)

    def test_missing_columns(self) -> None:
        prices = pd.DataFrame({"code": ["1111"]})
        result = ResearchPreflightResult()
        validate_prices_pit(prices, result)
        assert any(i.code == "PRICES_NOT_PIT" for i in result.issues)

    def test_invalid_timestamp(self) -> None:
        prices = pd.DataFrame({
            "code": ["1111"],
            "date": ["not-a-date"],
            "available_at": ["not-valid"],
        })
        result = ResearchPreflightResult()
        validate_prices_pit(prices, result)
        assert any(i.code == "PRICE_INVALID_TIMESTAMP" for i in result.issues)


class TestValidateTradeOutputs:
    def test_pass(self) -> None:
        trades = pd.DataFrame({
            "code": ["1111"],
            "pnl": [100.0],
        })
        result = ResearchPreflightResult()
        validate_trade_outputs(
            trades=trades,
            open_positions=pd.DataFrame(),
            rejected_orders=pd.DataFrame(),
            result=result,
        )
        assert result.passed

    def test_unresolved_positions(self) -> None:
        trades = pd.DataFrame({"code": ["1111"], "pnl": [100.0]})
        open_pos = pd.DataFrame({"code": ["2222"]})
        result = ResearchPreflightResult()
        validate_trade_outputs(trades=trades, open_positions=open_pos, rejected_orders=pd.DataFrame(), result=result)
        assert any(i.code == "UNRESOLVED_POSITIONS" for i in result.issues)

    def test_no_trades(self) -> None:
        result = ResearchPreflightResult()
        validate_trade_outputs(trades=pd.DataFrame(), open_positions=pd.DataFrame(), rejected_orders=pd.DataFrame(), result=result)
        assert any(i.code == "NO_TRADES" for i in result.issues)

    def test_forced_exit(self) -> None:
        trades = pd.DataFrame({
            "code": ["1111"],
            "pnl": [100.0],
            "forced_exit_reason": ["EXIT_DEFERRED"],
        })
        result = ResearchPreflightResult()
        validate_trade_outputs(trades=trades, open_positions=pd.DataFrame(), rejected_orders=pd.DataFrame(), result=result)
        assert any(i.code == "FORCED_EXIT_PRESENT" for i in result.issues)

    def test_system_rejection(self) -> None:
        trades = pd.DataFrame({"code": ["1111"], "pnl": [100.0]})
        rejected = pd.DataFrame({
            "code": ["1111"],
            "reason": ["INVALID_EQUITY_AT_OPEN"],
        })
        result = ResearchPreflightResult()
        validate_trade_outputs(trades=trades, open_positions=pd.DataFrame(), rejected_orders=rejected, result=result)
        assert any(i.code == "SYSTEM_REJECTION_PRESENT" for i in result.issues)


class TestRunResearchPreflight:
    def test_integration_pass(self) -> None:
        cfg = {
            "data": {
                "price_vintage_mode": "point_in_time",
                "source": "jquants",
            },
            "backtest": {
                "impact_k_is_calibrated": True,
                "require_corporate_actions": True,
            },
            "research": {
                "trial_registry_enabled": True,
                "allow_test_reuse": False,
                "split": {
                    "train_start": "2010-01-01",
                    "train_end": "2018-12-31",
                    "validation_start": "2019-01-01",
                    "validation_end": "2022-12-31",
                    "test_start": "2023-01-01",
                    "test_end": "2025-12-31",
                },
            },
        }

        universe = pd.DataFrame({
            "code": ["1111"],
            "effective_from": ["2010-01-01"],
            "effective_to": [pd.NA],
            "available_at": ["2010-01-01 08:00:00+00:00"],
        })

        prices = pd.DataFrame({
            "code": ["1111"],
            "date": ["2024-01-10"],
            "available_at": ["2024-01-10 08:00:00+00:00"],
        })

        trades = pd.DataFrame({"code": ["1111"], "pnl": [100.0]})

        result = run_research_preflight(
            cfg=cfg,
            universe=universe,
            prices=prices,
            trades=trades,
            open_positions=pd.DataFrame(),
            rejected_orders=pd.DataFrame(),
        )
        assert result.passed, f"unexpected issues: {result.issues}"

    def test_integration_fail(self) -> None:
        cfg = {
            "data": {"price_vintage_mode": "latest_snapshot", "source": "yfinance"},
            "backtest": {},
            "research": {},
        }

        universe = pd.DataFrame({"code": ["1111"]})
        trades = pd.DataFrame()

        result = run_research_preflight(
            cfg=cfg,
            universe=universe,
            prices=None,
            trades=trades,
            open_positions=pd.DataFrame(),
            rejected_orders=pd.DataFrame(),
        )
        assert not result.passed
        assert len(result.issues) >= 3
