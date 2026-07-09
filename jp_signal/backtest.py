"""バックテストエンジン（FR-BT-01〜06）。

約定・コストモデル:
  - FR-BT-01: 買い指値 P は当日安値 < P で約定。
  - FR-BT-02: commission_bp
  - FR-BT-03: 金利/貸株 年率を /365 で日次化（暦日近似）
  - FR-BT-04: impact_bp = k * sqrt(order_value / adv)
              adv は直近 adv_window 日の平均売買代金
  - FR-BT-05: 売りは shortability 確認済みのみ
  - FR-BT-06: 金利/貸株コストは実経過暦日数（calendar days）で按分
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED_SIGNAL_COLS = {
    "code",
    "date",
    "side",
    "qty",
    "order_type",
    "limit_price",
    "holding_days",
}
_VALID_SIDES = {"BUY", "SELL"}


class Backtester:
    def __init__(
        self,
        impact_k_bp: float = 30.0,
        annual_interest_rate: float = 0.02,
        annual_lending_rate: float = 0.02,
        commission_bp: float = 0.0,
        half_spread_bp: float = 0.0,
        require_liquidity_data: bool = True,
        adv_window: int = 20,
    ):
        self.k = float(impact_k_bp)
        self.annual_int = float(annual_interest_rate)
        self.annual_lend = float(annual_lending_rate)
        self.daily_int = self.annual_int / 365.0
        self.daily_lend = self.annual_lend / 365.0
        self.commission_bp = float(commission_bp)
        self.half_spread_bp = float(half_spread_bp)
        self.require_liquidity_data = bool(require_liquidity_data)
        self.adv_window = max(1, int(adv_window))

    # ------------------------------------------------------------ fill models
    @staticmethod
    def _fills_limit_buy(day_low: float, limit: float) -> bool:
        """買い指値: 安値 < 指値 で約定（同値未約定）。"""
        return day_low < limit

    @staticmethod
    def _fills_limit_sell(day_high: float, limit: float) -> bool:
        """売り指値: 高値 > 指値 で約定（同値未約定）。"""
        return day_high > limit

    # -------------------------------------------------------- market impact
    def market_impact_bp(self, order_value: float, adv: float) -> float:
        """sqrt則によるマーケットインパクト(bp)。adv<=0 のときは0扱い。"""
        if adv is None or adv <= 0 or order_value <= 0:
            return 0.0
        return self.k * float(np.sqrt(order_value / adv))

    def _slippage_bp(self, order_value: float, adv: float) -> float:
        """1回の約定にかかる不利方向スリッページ(bp)。"""
        return (
            self.market_impact_bp(order_value, adv)
            + self.half_spread_bp
            + self.commission_bp
        )

    def _apply_slippage(
        self, price: float, order_value: float, adv: float, direction: int
    ) -> float:
        """不利側に価格を寄せる。買い(direction=+1)は上振れ、売り(direction=-1)は下振れ。"""
        bp = self._slippage_bp(order_value, adv)
        return price * (1.0 + direction * bp / 10000.0)

    # --------------------------------------------------------------- helpers
    def _prev_adv(self, px: pd.DataFrame, code: str, d) -> float:
        """d より前の最大 adv_window 日平均売買代金。"""
        try:
            g = px.loc[code]
        except KeyError:
            return 0.0
        prev = g.loc[g.index < d]
        if prev.empty:
            return 0.0
        return float(prev.tail(self.adv_window)["turnover"].mean())

    def _get_future_row(self, px: pd.DataFrame, code: str, d, holding_days: int):
        """d から holding_days 営業日後の (行, その日付) を返す。"""
        try:
            g = px.loc[code]
        except KeyError:
            return None, None
        fut = g.loc[g.index > d]
        if len(fut) < holding_days:
            return None, None
        exit_row = fut.iloc[holding_days - 1]
        exit_date = fut.index[holding_days - 1]
        return exit_row, exit_date

    def _latest_shortability_before(
        self, sh: pd.DataFrame, code: str, d
    ) -> dict | None:
        """d 以前の直近の shortability スナップショットを返す。"""
        try:
            g = sh.loc[code]
        except KeyError:
            return None
        prev = g.loc[g.index <= d]
        if prev.empty:
            return None
        return dict(prev.iloc[-1])

    def _entry_price(self, row, sig, adv: float, side_sign: int):
        """エントリー約定価格を返す。約定しない場合は None。"""
        order_type = sig.get("order_type", "MKT_OPEN")
        qty = sig.get("qty", 0)

        if order_type == "LIMIT":
            lp = sig.get("limit_price", np.nan)
            if lp is None or (isinstance(lp, float) and np.isnan(lp)):
                return None
            if side_sign > 0:  # BUY
                if not self._fills_limit_buy(row["low"], lp):
                    return None
                base = min(lp, row["open"])
            else:  # SELL
                if not self._fills_limit_sell(row["high"], lp):
                    return None
                base = max(lp, row["open"])
        elif order_type == "MKT_CLOSE":
            base = row["close"]
        else:  # MKT_OPEN
            base = row["open"]

        order_value = float(base) * float(qty)
        return self._apply_slippage(float(base), order_value, adv, direction=side_sign)

    # ------------------------------------------------------------------- run
    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        shortability: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """バックテストを実行する。"""
        if signals is None or signals.empty or prices is None or prices.empty:
            return pd.DataFrame()

        missing_cols = _REQUIRED_SIGNAL_COLS - set(signals.columns)
        if missing_cols:
            raise ValueError(f"signals に必須列が不足: {missing_cols}")

        px = prices.copy()
        px["date"] = pd.to_datetime(px["date"])
        px = px.set_index(["code", "date"]).sort_index()

        if shortability is not None and not shortability.empty:
            sh = shortability.copy()
            sh["date"] = pd.to_datetime(sh["date"])
            sh = sh.set_index(["code", "date"]).sort_index()
        else:
            sh = None

        results: list[dict] = []
        for _, s in signals.iterrows():
            code = s["code"]
            d = pd.to_datetime(s["date"])

            if s["side"] not in _VALID_SIDES:
                results.append({**s.to_dict(), "status": "INVALID_SIDE"})
                continue

            qty = s.get("qty", 0)
            try:
                qty = int(qty)
            except (TypeError, ValueError):
                qty = 0
            if qty <= 0:
                results.append({**s.to_dict(), "status": "INVALID_QTY"})
                continue

            if (code, d) not in px.index:
                results.append({**s.to_dict(), "status": "NO_PRICE_DATA"})
                continue

            row = px.loc[(code, d)]
            adv = self._prev_adv(px, code, d)

            # 流動性データ不足チェック
            if self.require_liquidity_data and adv <= 0:
                results.append({**s.to_dict(), "status": "NO_ENTRY_LIQUIDITY_DATA"})
                continue

            # 売り可否チェック（FR-BT-05）
            if s["side"] == "SELL":
                snap = (
                    self._latest_shortability_before(sh, code, d)
                    if sh is not None
                    else None
                )
                shortable = (
                    snap is not None
                    and int(snap["is_margin_lendable"]) == 1
                    and int(snap["short_restricted"]) == 0
                )
                if not shortable:
                    results.append({**s.to_dict(), "status": "SKIP_NOT_SHORTABLE"})
                    continue

            side_sign = 1 if s["side"] == "BUY" else -1
            entry = self._entry_price(row, s, adv, side_sign)
            if entry is None:
                results.append({**s.to_dict(), "status": "NO_FILL"})
                continue

            holding_days = int(s.get("holding_days", 1))
            exit_row, exit_date = self._get_future_row(px, code, d, holding_days)
            if exit_row is None:
                results.append({**s.to_dict(), "status": "NO_EXIT_DATA"})
                continue

            exit_base = float(exit_row["close"])
            exit_adv = self._prev_adv(px, code, exit_date)
            exit_price = self._apply_slippage(
                exit_base, exit_base * qty, exit_adv, direction=-side_sign
            )

            # FR-BT-06: 実経過暦日数で按分
            cal_days = (exit_date - d).days
            cal_days = max(cal_days, 1)
            carry_rate = self.daily_lend if s["side"] == "SELL" else self.daily_int
            carry_cost = entry * carry_rate * cal_days

            direction = side_sign
            gross = (exit_price - entry) * direction * qty
            pnl = gross - carry_cost * qty

            results.append(
                {
                    **s.to_dict(),
                    "entry": entry,
                    "exit": exit_price,
                    "carry_cost_per_share": carry_cost,
                    "gross_pnl": gross,
                    "pnl": pnl,
                    "status": "FILLED",
                }
            )

        return pd.DataFrame(results)
