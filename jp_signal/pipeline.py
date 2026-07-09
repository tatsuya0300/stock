"""日次パイプライン統合（FR-DATA + FR-MODEL + FR-SIZE + FR-NOTIFY）。

寄前パイプライン: データ取得 → シグナル生成 → サイズ算定 → リスク制限 → 通知。
dry-run モードでは DB 書込と発注指示送信をスキップする。
戻り値として最終 orders を返す（テスト容易性）。
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, timedelta

import pandas as pd

from .calendar import is_tse_business_day
from .datasource import JQuantsSource, YFinanceSource
from .model import MeanReversionRule
from .notifier import ConsoleNotifier, DiscordNotifier, format_orders
from .risk import apply_order_risk_limits, risk_config_from_dict
from .sizing import compute_size
from .storage import Storage
from .universe import load_universe

log = logging.getLogger(__name__)


def _latest_shortable(
    shortability: pd.DataFrame, code: str, as_of: date
) -> bool:
    """指定コード・日付時点の最短過去の売り可否を返す。"""
    if shortability is None or shortability.empty:
        return False

    sub = shortability[shortability["code"] == code].copy()
    if sub.empty:
        return False

    sub["date"] = pd.to_datetime(sub["date"])
    past = sub[sub["date"] <= pd.Timestamp(as_of)]
    if past.empty:
        return False

    latest = past.sort_values("date").iloc[-1]
    return (
        int(latest["is_margin_lendable"]) == 1
        and int(latest["short_restricted"]) == 0
    )


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


def morning_pipeline(
    as_of: date, cfg: dict, dry_run: bool = False
) -> pd.DataFrame:
    """寄前発注指示を生成し通知する。戻り値は最終 orders。

    FR-BT-05: SELL orders are dropped unless shortability is confirmed.
    """
    if not is_tse_business_day(as_of):
        log.info("non-business day: %s", as_of)
        return pd.DataFrame()

    run_id = uuid.uuid4().hex[:12]
    storage: Storage | None = Storage(cfg["data"]["db_path"]) if not dry_run else None
    ds = make_datasource(cfg)
    univ = load_universe(cfg["universe"], as_of=str(as_of))
    codes = univ["code"].tolist()
    notifier = make_notifier(cfg)

    try:
        start = as_of - timedelta(days=400)
        # yfinance end は exclusive。as_of 当日を含めるため +1 日。
        end = as_of + timedelta(days=1)
        df = ds.fetch_daily(codes, start, end)
        if df.empty:
            notifier.send("本日はシグナル生成不可", "データ取得失敗")
            return pd.DataFrame()

        # as_of 超のデータが混入した場合は落とす
        df = df[df["date"] <= str(as_of)]
        if df.empty:
            notifier.send("本日はシグナル生成不可", "データ取得失敗(as_of超のみ)")
            return pd.DataFrame()

        if not dry_run and storage is not None:
            storage.upsert_prices(df)

        # shortability を DB から読み込み
        shortability_df = pd.DataFrame()
        if not dry_run and storage is not None:
            shortability_df = storage.load_shortability(
                codes,
                start=str(as_of - timedelta(days=30)),
                end=str(as_of),
            )

        prices = df
        model = MeanReversionRule(lookback=5, top_n=5)
        sig = model.generate(prices, as_of=str(as_of))
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

        # サイズ算定（前日終値・前日代金ベース）
        prev = prices[prices["date"] < str(as_of)].sort_values("date")
        last_row = prev.groupby("code").tail(1).set_index("code")
        univ_idx = univ.set_index("code")

        rows = []
        for _, r in sig.iterrows():
            code = r["code"]
            if code not in last_row.index:
                continue
            # ref_price: raw close (約定額計算用)
            ref = float(last_row.loc[code, "close"])
            turnover = float(last_row.loc[code, "turnover"])
            qty, yen, warn = compute_size(
                turnover,
                ref,
                cfg["sizing"]["adv_ratio"],
                cfg["sizing"]["adv_ratio_cap"],
                unit=100,
                market_open_unit_cap=cfg["sizing"]["market_open_unit_cap"],
                is_market_open_order=True,
            )
            if qty == 0:
                continue

            # 未取得・不明は売り不可（保守側）。FR-BT-05 と整合。
            shortable = (
                _latest_shortable(shortability_df, code, as_of)
                if not shortability_df.empty
                else False
            )
            name = univ_idx.loc[code, "name"] if code in univ_idx.index else ""
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "side": r["side"],
                    "order_type": "MKT_OPEN",
                    "qty": qty,
                    "ref_price": ref,
                    "value_yen": yen,
                    "score": float(r.get("score", 0)),
                    "warn": warn,
                    "shortable": shortable,
                }
            )

        orders = pd.DataFrame(rows)
        if orders.empty:
            notifier.send("本日はシグナル生成不可", "サイズ算定後に0件")
            return pd.DataFrame()

        # リスク制限を適用
        risk_cfg = risk_config_from_dict(cfg.get("risk", {}))
        orders = apply_order_risk_limits(orders, risk_cfg, score_col="score")

        if orders.empty:
            notifier.send("本日はシグナル生成不可", "リスク制限後に0件")
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
