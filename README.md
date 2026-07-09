# jp_signal — 日本株シグナル生成 + バックテスト MVP

要件定義書のうち **FR-DATA（データ取得基盤）** と **FR-BT（バックテスト）** の最小実装（MVP）。  
FR-NOTIFY はアダプタ差し替え設計（`ConsoleNotifier` 先行、Discord 対応済み）。

### 重要

本リポジトリは「発注指示の生成・通知」までを対象とする。自動発注・投資助言ではない。  
`MeanReversionRule` は動作確認用ダミーであり、収益性を保証しない。

## 設計方針

1. **言語/構成**: Python 3.11+ / モジュール分割 / 型ヒント。依存は最小。
2. **データソース**: プロトタイプは `yfinance`、本番は `JQuantsSource`（同一 IF `PriceDataSource`）。
3. **売り可否（FR-DATA-04）**: 日証金は最新スナップショットのみのため日次蓄積が必要。  
   現状 `snapshot_today()` は未実装。未確認売りはデフォルトで出さない。
4. **BT約定モデル（FR-BT-01）**: 買い指値 P → 当日安値 < P。
5. **マーケットインパクト（FR-BT-04）**: `impact_bp = k * sqrt(order_value / adv)`。エントリー/エグジット往復計上。
6. **信用金利/貸株料（FR-BT-03）**: 年率設定値を /365 で日次化。
7. **通知（FR-NOTIFY）**: Console / Discord アダプタ。
8. **look-ahead 回避**: 寄前は前営業日までの価格のみ使用。

### 一次情報

- JQuants API: https://jpx.gitbook.io/j-quants-ja / https://jpx-jquants.com/en/spec/eq-bars-daily
- 日本証券金融: https://www.jsf.co.jp/
- 東証 売買制度: https://www.jpx.co.jp/rules-participants/rules/regulations/
- TOPIX 指数: https://www.jpx.co.jp/markets/indices/topix/
- yfinance: https://github.com/ranaroussi/yfinance

## ディレクトリ構成（現行）

```
.
├── .github/workflows/ci.yml   # CI（ruff + mypy + pytest）
├── config.yaml
├── main.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── data/
│   └── topix500_sample.csv    # 動作確認用サンプル（デフォルト）
├── scripts/
│   ├── run_backtest.py
│   ├── build_pit_universe_from_events.py
│   ├── build_universe_from_jpx_weights.py
│   └── import_fills.py
├── tests/
└── jp_signal/
    ├── __init__.py
    ├── calendar.py
    ├── config.py
    ├── data_quality.py
    ├── datasource.py
    ├── shortability.py
    ├── storage.py
    ├── universe.py
    ├── model.py
    ├── sizing.py
    ├── risk.py
    ├── order_builder.py
    ├── backtest.py
    ├── metrics.py
    ├── notifier.py
    └── pipeline.py
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 開発時
```

## ユニバース CSV

| 用途 | ファイル | 設定 |
|---|---|---|
| 動作確認（デフォルト） | `data/topix500_sample.csv` | `universe.file: ./data/topix500_sample.csv` |
| 本格BT/運用 | `data/topix500.csv`（自前配置） | `universe.file: ./data/topix500.csv` |

### data/topix500.csv の用意手順（本格利用時）

1. JPX の TOPIX 系指数ページから構成銘柄（またはウェイトファイル）を取得  
   https://www.jpx.co.jp/markets/indices/topix/
2. 最低限 `code,name` の CSV を作成（証券コードは 4 桁想定）
3. 可能なら point-in-time 用に `effective_from,effective_to` を付与
4. `config.yaml` の `universe.file` を `./data/topix500.csv` に変更

**補助スクリプト:**
- `scripts/build_universe_from_jpx_weights.py`
- `scripts/build_pit_universe_from_events.py`（生存者バイアス対策の土台）

## 使い方

### 寄前パイプライン

```bash
python main.py
# または
python main.py morning
python main.py morning --dry-run
python main.py morning --date 2026-07-09
```

### 引け後

```bash
python main.py closing
python main.py closing --fills data/fills_YYYY-MM-DD.csv
```

### バックテスト

```bash
# 事前に価格データを DB へ取り込む（main.py 実行など）
python scripts/run_backtest.py
```

## データソース切替

| 項目 | yfinance（試作） | jquants（本番） |
|---|---|---|
| `data.source` | `yfinance` | `jquants` |
| 認証 | 不要 | 環境変数 `JQUANTS_API_KEY` |
| turnover | 近似 close*volume | 真値（Va/TurnoverValue） |
| sizing/impact | デフォルト拒否 | 利用可 |

### yfinance のまま sizing/impact を走らせる（非推奨・疎通確認のみ）:

```yaml
data:
  source: "yfinance"
  allow_approximate_turnover: true
```

### 本番例:

```bash
export JQUANTS_API_KEY=...   # V2 API Key
# config.yaml:
#   data.source: "jquants"
python main.py morning
```

### Discord 通知:

```bash
export DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
# config.yaml: notify.channel: "discord"
```

## 運用ルール（P0 強制）

1. **shortability 未実装のまま売りを出さない**  
   - `risk.allow_short_without_confirmed_shortability: false`（デフォルト）  
   - shortability データが空なら売りは全除外  
   - 開発で売りを通す場合のみ明示的に `true`（本番禁止）

2. **yfinance を本番 sizing/impact に使わない**  
   - `allow_approximate_turnover: false`（デフォルト）で hard fail  
   - 本番は `jquants`

3. **impact_k 未較正の PnL を過信しない**  
   - `impact_k_is_calibrated: false` の間は研究用

4. **自動執行しない**  
   - 通知は参考情報。最終判断・発注は人間

## 要件充足マッピング（要約）

| 要件ID | 実装箇所 | 備考 |
|---|---|---|
| FR-DATA-01/03/05 | datasource.py, storage.py | yfinance/JQuants 切替可 |
| FR-DATA-04 | shortability.py | スケルトン（未実装） |
| FR-UNIV-01/02/03 | universe.py | PIT列対応（データ次第） |
| FR-MODEL-01/02/04 | model.py | ダミールール |
| FR-BT-01〜06 | backtest.py, metrics.py | 往復コスト対応 |
| FR-SIZE-01/02 | sizing.py | adv cap 対応 |
| FR-RISK | risk.py, order_builder.py | 未確認売り除外 |
| FR-NOTIFY | notifier.py, pipeline.py | Console/Discord |
| NFR-OPS | calendar.py | jpholiday |

## 明示的な弱点（忖度なし）

1. `shortability.py` 本実装未了。売り戦略BT/運用は信頼できない。
2. yfinance turnover は近似。本番は JQuants 必須。
3. `MeanReversionRule` はダミー。収益性なし。
4. JQuants の列名・ページング等は公式仕様の再確認が必要。
5. `impact_k_bp` は未較正。
6. 生存者バイアス: point-in-time ユニバース実データが無いと未解消。
7. 本改訂は正しさ向上が目的であり、収益性を保証しない。

## 開発

```bash
ruff check .
mypy jp_signal --ignore-missing-imports
pytest -q
```

CI: `.github/workflows/ci.yml`（push / pull_request）

## 次の着手候補

1. `data/topix500.csv`（可能なら PIT）を用意し JQuants で BT
2. `shortability.py` の日証金取り込み本実装
3. fills から realized slippage を推定し `impact_k` を較正
4. ダミーモデル置換と walk-forward 評価
