"""日次パイプライン統合（FR-DATA + FR-MODEL + FR-SIZE + FR-NOTIFY）。

寄前パイプライン: データ取得 → シグナル生成 → サイズ算定 → 通知。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yaml

from .calendar import is_tse_business_day
from .datasource import JQuantsSource, YFinanceSource
from .model import MeanReversionRule
from .notifier import ConsoleNotifier, DiscordNotifier, format_orders
from .sizing import compute_size
from .storage import Storage
from .universe import load_universe


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_datasource(cfg: dict):
    if cfg["data"]["source"] == "jquants":
        return JQuantsSource(cfg["data"]["jquants_refresh_token"])
    return YFinanceSource()


def make_notifier(cfg: dict):
    ch = cfg["notify"]["channel"]
    if ch == "discord":
        return DiscordNotifier(cfg["notify"]["discord_webhook"])
    return ConsoleNotifier()


def morning_pipeline(as_of: date, cfg: dict) -> None:
    """寄前発注指示を生成し通知する。休場日は通知抑止（FR-NOTIFY）。"""
    if not is_tse_business_day(as_of):
        return

    storage = Storage(cfg["data"]["db_path"])
    ds = make_datasource(cfg)
    univ = load_universe(cfg["universe"]["file"])
    codes = univ["code"].tolist()
    notifier = make_notifier(cfg)

    # データ増分取得（過去約400日）
    start = as_of - timedelta(days=400)
    df = ds.fetch_daily(codes, start, as_of)
    if df.empty:
        notifier.send("本日はシグナル生成不可", "データ取得失敗")
        return
    storage.upsert_prices(df)

    prices = storage.load_prices(codes, str(start), str(as_of))
    model = MeanReversionRule(lookback=5, top_n=5)
    sig = model.generate(prices, as_of=str(as_of))
    if sig.empty:
        notifier.send("本日はシグナル生成不可", "シグナル0件")
        return

    # サイズ算定（前日終値・前日代金ベース）
    prev = prices[prices["date"] < str(as_of)].sort_values("date")
    last_row = prev.groupby("code").tail(1).set_index("code")
    univ_idx = univ.set_index("code")

    rows = []
    for _, r in sig.iterrows():
        code = r["code"]
        if code not in last_row.index:
            continue
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
                "warn": warn,
                "shortable": True,  # TODO: shortability 連携（MVP）
            }
        )

    orders = pd.DataFrame(rows)
    if orders.empty:
        notifier.send("本日はシグナル生成不可", "サイズ算定後に0件")
        return
    notifier.send(f"寄前発注指示 {as_of}", format_orders(orders))
