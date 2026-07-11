"""日次パイプライン統合（FR-DATA + FR-MODEL + FR-SIZE + FR-NOTIFY）。

寄前パイプライン: データ取得 → シグナル生成 → サイズ算定 → リスク制限 → 通知。
dry-run モードでは DB 書込と発注指示送信をスキップする。
戻り値として最終 orders を返す（テスト容易性）。

v2 変更点:
  - order_builder.signals_to_orders に注文生成を一元化（live/BT 同一ロジック）
  - 価格取得を差分更新に（storage に日付が揃っている分は再取得しない）
  - storage の context manager 使用

P0:
  - yfinance 近似 turnover の sizing/impact 利用をガード
  - shortability 未確認売りは risk 設定で禁止（デフォルト）
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .calendar import is_tse_business_day
from .config import enforce_short_policy_for_live, guard_approximate_turnover
from .coverage import CoverageThresholds, validate_daily_coverage
from .datasource import JQuantsSource, YFinanceSource
from .model import MeanReversionRule
from .notifier import ConsoleNotifier, DiscordNotifier, format_orders
from .order_builder import build_orders_with_audit
from .risk import risk_config_from_dict
from .storage import Storage
from .universe import load_universe

log = logging.getLogger(__name__)

# 初回取得時の過去日数（足りなければ随時追加）
_INITIAL_HISTORY_CALENDAR_DAYS = 400


def make_datasource(cfg: dict):
    if cfg["data"]["source"] == "jquants":
        return JQuantsSource(cfg["data"]["jquants_api_key"])
    return YFinanceSource()


def make_notifier(cfg: dict):
    ch = cfg["notify"]["channel"]
    if ch == "discord":
        webhook = cfg["notify"].get("discord_webhook", "")
        return DiscordNotifier(webhook)
    return ConsoleNotifier()


def _fetch_prices_incremental(
    ds,
    storage: Storage | None,
    codes: list[str],
    as_of: date,
    *,
    dry_run: bool = False,
) -> pd.DataFrame:
    """銘柄単位で価格を差分取得する。

    注意:
      - 新規銘柄は初期履歴を取得
      - 既存銘柄は銘柄ごとの最終日翌日から取得
      - 同じ取得開始日の銘柄はまとめてAPIへ渡す
    """
    history_start = as_of - timedelta(days=_INITIAL_HISTORY_CALENDAR_DAYS)

    normalized_codes = sorted({str(code) for code in codes})

    if not normalized_codes:
        return pd.DataFrame()

    if dry_run or storage is None:
        return ds.fetch_daily(
            normalized_codes,
            history_start,
            as_of,
        )

    existing = storage.load_prices(
        normalized_codes,
        start=history_start.isoformat(),
        end=as_of.isoformat(),
    )

    # start日ごとにコードをグループ化してAPI呼出回数を抑える
    fetch_groups: dict[date, list[str]] = {}

    for code in normalized_codes:
        code_rows = existing[existing["code"].astype(str) == code]

        if code_rows.empty:
            fetch_start = history_start
        else:
            last_date = pd.to_datetime(code_rows["date"]).max().date()
            fetch_start = last_date + timedelta(days=1)

        if fetch_start <= as_of:
            fetch_groups.setdefault(fetch_start, []).append(code)

    fetched_frames: list[pd.DataFrame] = []

    for fetch_start, group_codes in sorted(fetch_groups.items()):
        fresh = ds.fetch_daily(
            group_codes,
            fetch_start,
            as_of,
        )

        if fresh is None or fresh.empty:
            log.warning(
                "価格差分取得失敗: start=%s codes=%s",
                fetch_start,
                group_codes[:10],
            )
            continue

        storage.upsert_prices(fresh)
        fetched_frames.append(fresh)

    # upsert後にDBから再読込し、既存＋新規を一貫した形で返す
    combined = storage.load_prices(
        normalized_codes,
        start=history_start.isoformat(),
        end=as_of.isoformat(),
    )

    if combined.empty:
        return combined

    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")

    return (
        combined[combined["date"] <= as_of.isoformat()]
        .sort_values(["code", "date"])
        .drop_duplicates(["code", "date"], keep="last")
        .reset_index(drop=True)
    )


def morning_pipeline(as_of: date, cfg: dict, dry_run: bool = False) -> pd.DataFrame:
    """寄前発注指示を生成し通知する。戻り値は最終 orders。"""
    if not is_tse_business_day(as_of):
        log.info("non-business day: %s", as_of)
        return pd.DataFrame()

    # P0: yfinance 近似を sizing に使う処理を拒否（明示オプトイン以外）
    guard_approximate_turnover(cfg, context="morning_pipeline")
    enforce_short_policy_for_live(cfg)

    run_id = uuid.uuid4().hex[:12]
    storage: Storage | None = Storage(cfg["data"]["db_path"]) if not dry_run else None
    ds = make_datasource(cfg)
    univ = load_universe(cfg["universe"], as_of=str(as_of))
    codes = univ["code"].tolist()
    notifier = make_notifier(cfg)

    # yfinance 近似の再警告（本番誤用防止）
    if cfg.get("data", {}).get("source") == "yfinance":
        log.warning("data.source=yfinance: turnover は近似。本番は jquants に切替必須。")
    if not cfg.get("backtest", {}).get("impact_k_is_calibrated", False):
        log.info(
            "impact_k_bp=%.1f は未較正（impact_k_is_calibrated=false）",
            float(cfg.get("backtest", {}).get("impact_k_bp", 30.0)),
        )

    try:
        df = _fetch_prices_incremental(ds, storage, codes, as_of, dry_run=dry_run)
        if df.empty:
            notifier.send("本日はシグナル生成不可", "データ取得失敗")
            return pd.DataFrame()

        df = df[df["date"] <= str(as_of)]
        if df.empty:
            notifier.send("本日はシグナル生成不可", "データ取得失敗(as_of超のみ)")
            return pd.DataFrame()

        # coverage check
        dq_cfg = cfg.get("data_quality", {})
        thresholds = CoverageThresholds.from_config(dq_cfg)
        coverage = validate_daily_coverage(
            df,
            univ,
            as_of=as_of,
            lookback=int(cfg.get("model", {}).get("lookback", 5)),
            adv_window=int(cfg.get("backtest", {}).get("adv_window", 20)),
            min_adv_periods=int(cfg.get("backtest", {}).get("min_adv_periods", 20)),
            thresholds=thresholds,
        )
        if not coverage.ok:
            log.warning(coverage.to_message())
            if thresholds.hard_fail:
                notifier.send(
                    "本日はシグナル生成中止",
                    f"coverage不足: {', '.join(coverage.failed_reasons)}",
                )
                return pd.DataFrame()

        # shortability を DB から読み込み
        shortability_df = pd.DataFrame()
        if not dry_run and storage is not None:
            shortability_df = storage.load_shortability(
                codes,
                start=str(as_of - timedelta(days=30)),
                end=str(as_of),
            )

        model = MeanReversionRule(
            lookback=int(cfg.get("model", {}).get("lookback", 5)),
            top_n=int(cfg.get("model", {}).get("top_n", 5)),
        )
        sig = model.generate(df, as_of=str(as_of))
        if sig.empty:
            notifier.send("本日はシグナル生成不可", "シグナル0件")
            return pd.DataFrame()

        if not dry_run and storage is not None:
            storage.append_signals(
                run_id=run_id,
                signals=sig,
                signal_asof_date=str(as_of),
                model_name=type(model).__name__,
            )

        risk_cfg = risk_config_from_dict(cfg.get("risk", {}))
        unit = int(cfg.get("sizing", {}).get("unit", 100))

        order_result = build_orders_with_audit(
            sig,
            df,
            as_of=as_of,
            sizing_cfg=cfg["sizing"],
            risk_cfg=risk_cfg,
            shortability=shortability_df if not shortability_df.empty else None,
            universe=univ,
            unit=unit,
            order_type="MKT_OPEN",
            for_backtest=False,
            shortability_max_age_days=int(
                cfg.get("risk", {}).get(
                    "shortability_max_age_days",
                    4,
                )
            ),
        )

        orders = order_result.selected

        if not dry_run and storage is not None and not order_result.rejected.empty:
            storage.append_order_rejections(
                run_id=run_id,
                rejection_date=str(as_of),
                rejected=order_result.rejected,
            )

        log.info(
            "order build diagnostics: %s",
            order_result.diagnostics,
        )

        if orders.empty:
            notifier.send("本日はシグナル生成不可", "サイズ算定・リスク制限後に0件")
            return pd.DataFrame()

        # P0: 売り注文が残っているのに shortability が取込済みで無い場合に警告
        n_sell = len(orders[orders["side"] == "SELL"]) if "side" in orders.columns else 0
        if n_sell > 0 and shortability_df.empty:
            log.error(
                "shortability 未取込なのに売り注文 %d 件が残存。"
                " risk.allow_short_without_confirmed_shortability を確認すること。",
                n_sell,
            )

        if not dry_run and storage is not None:
            orders["order_date"] = str(as_of)
            orders["signal_asof_date"] = str(as_of)
            storage.append_orders(run_id, orders)

        if dry_run:
            notifier.send(f"[DRY-RUN] 寄前発注指示 {as_of}", format_orders(orders))
        else:
            notifier.send(f"寄前発注指示 {as_of}", format_orders(orders))

        return orders

    finally:
        if not dry_run and storage is not None:
            storage.close()


def closing_pipeline(
    as_of: date,
    cfg: dict,
    fills_csv: str | None = None,
    dry_run: bool = False,
) -> dict:
    """引け後処理。

    1) 当日 orders の再通知（確認用）
    2) 任意の fills CSV を DB へ取込（FR-RECORD 最小実装）
    3) 当日 fills 件数を通知

    fills CSV 列: trade_date,code,side,qty,price[,note,run_id]
    """
    if not is_tse_business_day(as_of):
        log.info("non-business day: %s", as_of)
        return {"orders": 0, "fills_imported": 0}

    storage: Storage | None = Storage(cfg["data"]["db_path"]) if not dry_run else None
    notifier = make_notifier(cfg)
    as_of_s = str(as_of)
    result = {"orders": 0, "fills_imported": 0}

    try:
        orders_df = pd.DataFrame()
        if storage is not None:
            orders_df = storage.load_orders(order_date=as_of_s)
            result["orders"] = len(orders_df)

        if orders_df.empty:
            body = "当日注文なし（または DB 未接続）"
        else:
            body = format_orders(orders_df)

        title_prefix = "[DRY-RUN] " if dry_run else ""
        notifier.send(f"{title_prefix}引け後確認 {as_of}", body)

        # fills 取込
        if fills_csv is not None and storage is not None and not dry_run:
            n = storage.import_fills_csv(fills_csv)
            result["fills_imported"] = n
            fills_today = storage.load_fills(trade_date=as_of_s)
            notifier.send(
                f"実績取込 {as_of}",
                f"CSV取込: {n}件\n当日fills: {len(fills_today)}件",
            )
        elif fills_csv is not None and dry_run:
            path = Path(fills_csv)
            n_lines = 0
            if path.exists():
                # ヘッダ除く概算
                n_lines = max(sum(1 for _ in path.open(encoding="utf-8")) - 1, 0)
            result["fills_imported"] = n_lines
            notifier.send(
                f"[DRY-RUN] 実績取込 {as_of}",
                f"CSV行数(概算): {n_lines}",
            )

        return result
    finally:
        if storage is not None:
            storage.close()
