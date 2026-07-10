## Summary

バックテストと運用パイプラインの最低限の正確性・再現性を改善しました。

## Changes

- GitHub Actions CIを実体化
- `main.py` と `scripts/run_backtest.py` に `--config` を追加
- demo / production 設定を分離
- BTで `allow_unconfirmed_short_in_bt` が実際に効くように修正
- サイジングを前日売買代金ではなく rolling ADV ベースに変更
- `run_backtest.py` の不要な `all_dates[20:]` を削除
- shortability policy と rolling ADV の回帰テストを追加

## Motivation

現状では以下の問題がありました。

- README上はCIがあるが、実際には `.github/workflows/ci.yml` が存在しない
- デフォルト設定のままでは `python main.py --dry-run` がhard failする
- `allow_unconfirmed_short_in_bt=true` でも Backtester 側でSELLが除外される
- サイジングにADVではなく前日単日のturnoverを使っていた
- warm-up期間を確保しているにもかかわらず、BT開始後さらに20営業日を捨てていた

## Scope

このPRは最小修正です。以下は別PR対象です。

- portfolio-level backtester
- cash / position / NAV ledger
- corporate action対応
- shortability ETL本実装
- DB migration framework
- fills重複防止

## Test

```bash
ruff check .
mypy jp_signal --ignore-missing-imports
pytest -q
```

## Notes

- `configs/demo.yaml` は疎通確認用です
- 本番・本格BTでは `configs/production.yaml` と J-Quants を使用してください
- 本PRの対象ブランチは `fix/backtest-ci-adv-short-policy` です
