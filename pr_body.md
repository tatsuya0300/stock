## P0対応 変更サマリ

| 項目 | 内容 |
|---|---|
| README | 現行ツリー・起動・ユニバース取得・運用ルール・弱点を同期 |
| デフォルトユニバース | topix500_sample.csv |
| CI | .github/workflows/ci.yml 正式配置 |
| yfinanceガード | guard_approximate_turnover() で sizing/impact hard fail |
| shortability | デフォルト未確認売り禁止を設定・BT・テストで固定化 |

### 変更ファイル一覧

1. config.yaml -- デフォルトを topix500_sample.csv に変更。allow_approximate_turnover: false を明示。shortability 未確認売り禁止をデフォルト化。
2. jp_signal/config.py -- guard_approximate_turnover, enforce_short_policy_for_live, uses_approximate_turnover を追加。環境変数による秘密情報上書き、各種バリデーション追加。
3. jp_signal/pipeline.py -- morning_pipeline に yfinance 近似ガードと shortpolicy 警告を追加。
4. scripts/run_backtest.py -- yfinance 近似ガード追加。shortability 未確認売りは BT でもデフォルト除外。
5. .github/workflows/ci.yml -- 新規作成。ruff + mypy + pytest を push/PR 時に実行。
6. README.md -- 現行ディレクトリ構成、起動手順、ユニバース取得手順、運用ルール、弱点を同期。
7. tests/test_config.py -- 各種バリデーションのテスト追加。
8. tests/test_short_policy.py -- 新規。未確認売り除外と確認済み売り許可のテスト。
