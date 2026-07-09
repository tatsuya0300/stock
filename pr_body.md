Summary

This PR fixes major research-correctness and production-safety issues in the Japanese equity signal MVP.

Main Changes

Research correctness

- Fixes look-ahead bias (model no longer uses as_of date's closing price).
- Separates signal_asof_date and actual order date.
- Uses raw OHLC for execution and adjusted OHLC for features/returns.
- ADV changed from 1-day turnover to 20-day average (`adv_window`).
- Skips trades with missing liquidity data by default.
- Uses calendar-day carry cost when enabled.

Data model (v3 schema)

- Adds raw and adjusted OHLC columns.
- Adds basic schema migration (v1/v2 → v3 backward compatible).
- Adds PRIMARY KEY on signals/orders tables for idempotent UPSERT.
- Replaces INSERT OR REPLACE with ON CONFLICT DO UPDATE everywhere.
- Adds shortability table with upsert/load methods.
- Adds orders, signals, and fills tables for auditability.

Live pipeline safety

- SELL orders are dropped unless shortability is confirmed (FR-BT-05).
- Missing shortability is treated as not shortable.
- Risk limits fully connected: max_orders_per_day, max_gross_exposure, max_single_name_exposure, max_long/short_exposure.
- Adds dry-run mode.
- Adds structured logging.
- yfinance `end` is exclusive; pipeline passes `as_of + 1 day`.
- Error notification on pipeline failure.

Configuration

- Environment variable override for JQUANTS_API_KEY (V2,推奨), JQUANTS_REFRESH_TOKEN (V1互換), and DISCORD_WEBHOOK.
- Validates adv_ratio > 0, adv_ratio <= adv_ratio_cap.
- Validates J-Quants API key required when source=jquants.
- Risk section with defaults.
- backtest section with adv_window, holding_days, etc.

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
- yfinance raw/adjusted schema normalization
- Fallback when Adj Close is missing
- J-Quants response schema validation
- Model uses adj_close and avoids look-ahead bias
- Shortability helper ignores future snapshots
- Pipeline helpers (_latest_shortable)

Scripts

- scripts/run_backtest.py: end-to-end backtest script using DB + config.
- scripts/build_universe_from_jpx_weights.py: JPX公開ウェイトファイルからユニバースCSV生成。
- scripts/build_pit_universe_from_events.py: ADD/REMOVEイベント履歴からPITユニバース構築。

Behavioral Changes

Backtest results may worsen because several optimistic assumptions were removed:

- Same-day close-to-open look-ahead is removed.
- ADV changed from single-day snapshot to 20-day average.
- Missing liquidity data no longer implies zero impact.
- SELL trades require confirmed shortability.
- Carry cost may be counted by calendar days.

This is intentional.

Limitations

This PR does not fully solve data availability issues that require external datasets:

- Historical point-in-time TOPIX500 membership must be supplied as data.
- JSF shortability ingestion still depends on official/public format confirmation.
- Reverse stock lending fee / gyaku-hibu requires additional data.
- TOPIX500 厳密な日次PITは未解決（イベント積み上げ or 有料マスタが必要）。
- --top 500 による近似は公式TOPIX500選定ロジックと一致しない。
- shortability 本実装は未着手。
- J-Quants レート制限（Free=5req/min）は sleep 緩和のみ。
- Existing SQLite DBs with PRIMARY KEY-less signals/orders tables should be deleted and recreated.

Test Plan

```bash
python -m pytest
python -m ruff check .
python -m mypy jp_signal
```
