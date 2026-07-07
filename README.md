# jp_signal — 日本株シグナル生成 + バックテスト MVP

要件定義書のうち **FR-DATA（データ取得基盤）** と **FR-BT（バックテスト）** の最小実装（MVP）。
FR-NOTIFY は後段でアダプタを差し替える設計（`ConsoleNotifier` 先行実装）。

## 設計方針

1. **言語/構成**: Python 3.11+ / モジュール分割 / 型ヒント付き。依存は最小（pandas, numpy, requests, pyyaml, python-dateutil, jpholiday, yfinance）。
2. **データソース**: プロトタイプは `yfinance`、本番は `JQuantsSource`（同一インターフェース `PriceDataSource`）で差し替え可能。
3. **売り可否（FR-DATA-04）**: 日証金は最新スナップショットのみ取得可のため、日次スナップショットを蓄積してヒストリカル化。過去分は「不明扱い」で売り戦略BTから除外。
4. **BT約定モデル（FR-BT-01）**: 買い指値 P → 当日安値 < P で約定（同値未約定）。売り指値 P → 当日高値 > P で約定。
5. **マーケットインパクト（FR-BT-04）**: `impact_bp = k * sqrt(order_value / adv)` の平方根則。`config.yaml` の `impact_k_bp` で較正。
6. **信用金利/貸株料（FR-BT-03）**: 年率2%（日次 = 2%/365）、オーバーナイト保有日数で計上、日計り0。
7. **通知（FR-NOTIFY）**: アダプタパターン。`ConsoleNotifier` 先行、Discord/Slack/メールは同一IFで後付け。
8. **ハルシネーション回避**: JQuants の詳細仕様は変動しうるため、確認箇所をコード内コメントで明示。yfinance は動作確認済みAPIのみ使用。

### 一次情報

- JQuants API 仕様: https://jpx.gitbook.io/j-quants-ja
- 日本証券金融（貸借銘柄・制限措置）: https://www.jsf.co.jp/
- 東証 売買制度: https://www.jpx.co.jp/rules-participants/rules/regulations/
- yfinance: https://github.com/ranaroussi/yfinance

## ディレクトリ構成

```
webapp/
├── config.yaml
├── requirements.txt
├── main.py
├── scripts/
│   └── run_backtest.py
├── data/
│   └── topix500_sample.csv   # 動作確認用サンプル
└── jp_signal/
    ├── __init__.py
    ├── calendar.py       # 祝日・営業日判定
    ├── datasource.py     # PriceDataSource + yfinance/JQuants 実装
    ├── shortability.py   # 売り可否（日証金）スナップショット管理
    ├── storage.py        # SQLite 永続化
    ├── universe.py       # TOPIX500 ユニバース管理
    ├── model.py          # シグナル生成モデル（ルールベース差替可）
    ├── sizing.py         # 執行サイズ算定
    ├── backtest.py       # バックテストエンジン
    ├── notifier.py       # 通知アダプタ
    └── pipeline.py       # 日次パイプライン統合
```

## 使い方（MVP）

```bash
pip install -r requirements.txt
mkdir -p data
# data/topix500.csv を用意（JPX公式のTOPIX500構成銘柄, code,name）
#   動作確認だけなら config.yaml の universe.file を data/topix500_sample.csv に向ける
python main.py
```

初回は yfinance から過去データを取得するため時間を要します。
cron / タスクスケジューラで平日 8:15 に `python main.py` を実行すると
`ConsoleNotifier` に発注指示が出ます。Discord へ切替は `config.yaml` の
`notify.channel: "discord"` と `discord_webhook` を設定するだけです。

### バックテスト

```bash
python scripts/run_backtest.py
```

## 要件充足マッピング

| 要件ID | 実装箇所 | 備考 |
|---|---|---|
| FR-DATA-01/03/05 | datasource.py, storage.py | yfinance/JQuants 切替可 |
| FR-DATA-02 | JQuantsSource 注記 | 直近2週間除外は取得側で日付制限 |
| FR-DATA-04 | shortability.py | スケルトン（一次情報要確認） |
| FR-UNIV-01/02/03 | universe.py, pipeline.py | TOPIX500 想定 |
| FR-MODEL-01/02/04 | model.py | ML は同IFで差替可 |
| FR-BT-01〜05 | backtest.py | インパクトは sqrt則で較正可 |
| FR-SIZE-01/02 | sizing.py | 50単元超は警告文字列で通知 |
| FR-NOTIFY-01〜06 | notifier.py, pipeline.py | Console/Discord アダプタ |
| FR-COMP | notifier.py: COMPLIANCE_FOOTER | 定型文で常時付与 |
| NFR-OPS | calendar.py | jpholiday 使用 |

## 明示的な弱点（忖度なし）

1. `shortability.py` は本実装未了。日証金の公開データフォーマット確認が必要。埋まるまで売り戦略BTは信頼できない。
2. yfinance の turnover は `close*volume` の近似。本番は必ず JQuants の `TurnoverValue` を使用。
3. `MeanReversionRule` は動作確認用ダミー。収益性は担保しない。実運用前に必ず BT で検証。
4. JQuants のエンドポイント名・パラメータは変更されうるため、`JQuantsSource` は公式ドキュメントで最新版を確認してから本番接続すること。
5. インパクト係数 `k=30bp` は較正例。実測値が溜まるまで暫定値。

## 次の着手候補

1. `data/topix500.csv` を用意し yfinance で日足を取り込み、1年分を BT → PnL・約定率・スリッページ分布を確認
2. `shortability.py` の日証金取り込み実装（一次情報確認要）
3. `DiscordNotifier` の実運用配線と cron スケジュール
4. FR-RECORD の実績入力（最小 CSV 追記でも可）
