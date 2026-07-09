"""日次パイプライン統合（FR-DATA + FR-MODEL + FR-SIZE + FR-NOTIFY）。

寄前パイプライン: データ取得 → シグナル生成 → サイズ算定 → リスク制限 → 通知。
dry-run モードでは DB 書込と発注指示送信をスキップする。
戻り値として最終 orders を返す（テスト容易性）。

v2 変更点:
  - order_builder.signals_to_orders に注文生成を一元化（live/BT 同一ロジック）
  - 価格取得を差分更新に（storage に日付が揃っている分は再取得しない）
  - storage の context manager 使用
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .calendar import is_tse_business_day
from .datasource import JQuantsSource, YFinanceSource
from .model import MeanReversionRule
from .notifier import ConsoleNotifier, DiscordNotifier, format_orders
from .order_builder import signals_to_orders
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
    """DB にあれば差分取得、無ければ初期ヒストリを取得。"""
    if dry_run or storage is None:
        start = as_of - timedelta(days=_INITIAL_HISTORY_CALENDAR_DAYS)
        # datasource 側で inclusive に正規化される前提
        return ds.fetch_daily(codes, start, as_of)

    # 既存の最終日を確認
    existing = storage.load_prices(
        codes,
        start=(as_of - timedelta(days=_INITIAL_HISTORY_CALENDAR_DAYS)).isoformat(),
        end=as_of.isoformat(),
    )

    if existing.empty:
        start = as_of - timedelta(days=_INITIAL_HISTORY_CALENDAR_DAYS)
        fresh = ds.fetch_daily(codes, start, as_of)
        if not fresh.empty:
            storage.upsert_prices(fresh)
        return fresh

    last_date = pd.to_datetime(existing["date"]).max().date()
    if last_date >= as_of:
        # 当日まで揃っている
        return existing[existing["date"] <= str(as_of)]

    # 不足分のみ取得
    start = last_date + timedelta(days=1)
    fresh = ds.fetch_daily(codes, start, as_of)
    if not fresh.empty:
        storage.upsert_prices(fresh)
    combined = pd.concat([existing, fresh], ignore_index=True)
    return combined[combined["date"] <= str(as_of)]


def morning_pipeline(
    as_of: date, cfg: dict, dry_run: bool = False
) -> pd.DataFrame:
    """寄前発注指示を生成し通知する。戻り値は最終 orders。"""
    if not is_tse_business_day(as_of):
        log.info("non-business day: %s", as_of)
        return pd.DataFrame()

    run_id = uuid.uuid4().hex[:12]
    storage: Storage | None = Storage(cfg["data"]["db_path"]) if not dry_run else None
    ds = make_datasource(cfg)
    univ = load_universe(cfg["universe"], as_of=str(as_of))
    codes = univ["code"].tolist()
    notifier = make_notifier(cfg)

    # yfinance 近似の再警告（本番誤用防止）
    if cfg.get("data", {}).get("source") == "yfinance":
        log.warning(
            "data.source=yfinance: turnover は近似。本番は jquants に切替推奨。"
        )
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

        orders = signals_to_orders(
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
        )

        if orders.empty:
            notifier.send("本日はシグナル生成不可", "サイズ算定・リスク制限後に0件")
            return pd.DataFrame()

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
