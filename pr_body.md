Summary

This PR fixes major research-correctness and production-safety issues in the Japanese equity signal MVP.

Main Changes

Research correctness

- Fixes look-ahead bias.
- Separates signal_asof_date and actual order date.
- Uses raw OHLC for execution and adjusted OHLC for features/returns.
- Skips trades with missing liquidity data by default.
- Uses calendar-day carry cost when enabled.

Data model

- Adds raw and adjusted OHLC columns.
- Adds basic schema migration.
- Replaces INSERT OR REPLACE with ON CONFLICT DO UPDATE.
- Adds orders, signals, and fills tables for auditability.

Live pipeline safety

- SELL orders are dropped unless shortability is confirmed.
- Missing shortability is treated as not shortable.
- Adds risk limits:
  - max orders per day
  - max gross exposure
  - max single-name exposure
  - max long/short exposure
- Adds dry-run mode.
- Adds process lock to prevent duplicate cron execution.
- Adds structured logging.

Universe

- Normalizes Japanese stock codes.
- Rejects duplicate codes.
- Supports point-in-time universe CSV with effective_from and effective_to.

Notifications

- Validates Discord webhook.
- Splits long Discord messages.

Tests

Adds regression tests for:

- Backtest fill logic
- Shortability behavior
- Missing liquidity handling
- Sizing validation
- Universe normalization and point-in-time filtering
- Risk limits
- Price data quality checks

Additional datasource tests:

- yfinance raw/adjusted schema normalization
- Fallback behavior when Adj Close is missing
- J-Quants response schema validation
- Datasource fetch wrapper behavior
- Model uses adj_close for returns
- Shortability helper ignores future snapshots

Behavioral Changes

Backtest results may worsen because several optimistic assumptions were removed:

- Same-day close-to-open look-ahead is removed.
- Missing liquidity data no longer implies zero impact.
- SELL trades require confirmed shortability.
- Carry cost may be counted by calendar days.

This is intentional.

Limitations

This PR does not fully solve data availability issues that require external datasets:

- Historical point-in-time TOPIX500 membership must be supplied as data.
- JSF shortability ingestion still depends on official/public format confirmation.
- Reverse stock lending fee / gyaku-hibu requires additional data.
- J-Quants column mapping should be verified against current official API documentation before production use.

Test Plan

```bash
python -m pytest
python -m ruff check .
```
