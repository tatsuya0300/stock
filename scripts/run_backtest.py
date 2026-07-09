"""バックテスト実行スクリプト。

使い方: python scripts/run_backtest.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jp_signal.backtest import Backtester
from jp_signal.calendar import previous_business_day
from jp_signal.config import load_config
from jp_signal.model import MeanReversionRule
from jp_signal.risk import apply_order_risk_limits, risk_config_from_dict
from jp_signal.sizing import compute_size
from jp_signal.storage import Storage
from jp_signal.universe import load_universe


def main() -> None:
    cfg = load_config()
    st = Storage(cfg["data"]["db_path"])
    try:
        univ_all = load_universe(cfg["universe"])
        codes = univ_all["code"].tolist()

        prices = st.load_prices(
            codes, cfg["backtest"]["start"], cfg["backtest"]["end"]
        )
        short = st.load_shortability(
            codes, cfg["backtest"]["start"], cfg["backtest"]["end"]
        )
        if prices.empty:
            print("価格データが空です。先に main.py 等で DB へ取り込んでください。")
            return

        model = MeanReversionRule(lookback=5, top_n=5)
        holding_days = int(cfg["backtest"].get("holding_days", 1))
        all_dates = sorted(prices["date"].unique())

        signal_frames: list[pd.DataFrame] = []
        for d in all_dates[20:]:
            # point-in-time ユニバース（CSV に effective_from/to がある場合のみ有効）
            univ_d = load_universe(cfg["universe"], as_of=d)
            codes_d = set(univ_d["code"].tolist())
            px_d = prices[prices["code"].isin(codes_d)]
            if px_d.empty:
                continue

            sig = model.generate(px_d, as_of=d)
            if sig.empty:
                continue

            # 寄前想定: 当日終値は未確定 → 前営業日までのデータで参照価格を計算
            as_of_ts = pd.Timestamp(d)
            cutoff = previous_business_day(as_of_ts.date()).isoformat()
            prev = px_d[px_d["date"] <= cutoff].sort_values("date")
            last_row = prev.groupby("code").tail(1).set_index("code")

            day_rows = []
            for _, r in sig.iterrows():
                code = r["code"]
                if code not in last_row.index:
                    continue
                ref = float(last_row.loc[code, "close"])
                adv_val = float(last_row.loc[code, "turnover"])
                qty, yen, warn = compute_size(
                    adv_val,
                    ref,
                    cfg["sizing"]["adv_ratio"],
                    cfg["sizing"]["adv_ratio_cap"],
                    unit=100,
                    market_open_unit_cap=cfg["sizing"]["market_open_unit_cap"],
                    is_market_open_order=True,
                )
                if qty == 0:
                    continue
                day_rows.append(
                    {
                        "code": code,
                        "date": d,
                        "side": r["side"],
                        "qty": qty,
                        "order_type": "MKT_OPEN",
                        "limit_price": None,
                        "holding_days": holding_days,
                        "value_yen": yen,
                        "score": float(r.get("score", 0)),
                        # BTでも shortability 未確認は不可扱い
                        "shortable": False if short.empty else None,
                    }
                )

            if not day_rows:
                continue

            day_df = pd.DataFrame(day_rows)

            # shortability 付与
            if not short.empty:
                sh = short.copy()
                sh["date"] = pd.to_datetime(sh["date"])
                asof_ts = pd.Timestamp(d)

                def _is_ok(code: str, _sh=sh, _asof=asof_ts) -> bool:
                    g = _sh[(_sh["code"] == code) & (_sh["date"] <= _asof)]
                    if g.empty:
                        return False
                    latest = g.sort_values("date").iloc[-1]
                    return (
                        int(latest["is_margin_lendable"]) == 1
                        and int(latest["short_restricted"]) == 0
                    )

                day_df["shortable"] = day_df["code"].apply(_is_ok)

            # リスク制限を適用（live pipeline と同じロジック）
            risk_cfg = risk_config_from_dict(cfg.get("risk", {}))
            day_df = apply_order_risk_limits(day_df, risk_cfg, score_col="score")

            if not day_df.empty:
                signal_frames.append(day_df)

        if not signal_frames:
            print("シグナル0件。バックテストをスキップします。")
            return

        signals = pd.concat(signal_frames, ignore_index=True)
        bt = Backtester(
            impact_k_bp=float(cfg["backtest"].get("impact_k_bp", 30.0)),
            annual_interest_rate=float(cfg["backtest"].get("annual_interest_rate", 0.02)),
            annual_lending_rate=float(cfg["backtest"].get("short_lending_rate", 0.02)),
            adv_window=int(cfg["backtest"].get("adv_window", 20)),
            require_liquidity_data=True,
        )
        result = bt.run(signals, prices, shortability=short if not short.empty else None)
        if result.empty:
            print("バックテスト結果が空です。")
            return

        filled = result[result["status"] == "FILLED"].copy()
        print(f"全シグナル: {len(result)}")
        print(f"約定: {len(filled)}")
        print(f"約定率: {(len(filled) / max(len(result), 1)) * 100:.1f}%")
        if not filled.empty:
            print(f"合計PnL: {filled['pnl'].sum():.0f}")
            print(f"勝率: {(filled['pnl'] > 0).mean() * 100:.1f}%")
            daily = filled.groupby("date")["pnl"].sum()
            print(f"日次PnL平均  : {daily.mean():.0f}")
            print(f"日次PnL標準偏差: {daily.std():.0f}")
    finally:
        st.close()


if __name__ == "__main__":
    main()
