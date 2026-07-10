"""バックテストエンジン（FR-BT-01〜06）。

約定・コストモデル:
  - FR-BT-01: 買い指値 P は当日安値 < P で約定（同値未約定）。
  - FR-BT-02: commission_bp / half_spread_bp
  - FR-BT-03: 金利/貸株 年率を /365 で日次化（暦日近似）
  - FR-BT-04: impact_bp = k * sqrt(order_value / adv)
              adv は直近 adv_window 日の平均売買代金
  - FR-BT-05: 売りは shortability 確認済みのみ
  - FR-BT-06: holding_days=1 は「翌営業日 close 決済」のオーバーナイト。
              zero_carry_for_intraday=True のとき carry=0（厳密な日計りではない）。
              複数日は実経過暦日数で carry を計上。
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
        zero_carry_for_intraday: bool = True,
        require_confirmed_shortability: bool = True,
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
        self.zero_carry_for_intraday = bool(zero_carry_for_intraday)
        self.require_confirmed_shortability = bool(require_confirmed_shortability)

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
    @staticmethod
    def _is_valid_adv(adv: float | None) -> bool:
        if adv is None:
            return False
        try:
            value = float(adv)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(value) and value > 0)

    def market_impact_bp(self, order_value: float, adv: float) -> float:
        if not np.isfinite(order_value) or order_value <= 0:
            raise ValueError(f"invalid order_value: {order_value}")

        if not self._is_valid_adv(adv):
            raise ValueError(f"invalid ADV: {adv}")

        participation = order_value / float(adv)
        return self.k * float(np.sqrt(participation))

    def _slippage_components(
        self, order_value: float, adv: float
    ) -> tuple[float, float, float, float]:
        """(impact_bp, half_spread_bp, commission_bp, total_bp)

        ADV が無効でも spread/commission は適用する。
        """
        if self._is_valid_adv(adv):
            impact = self.market_impact_bp(order_value, adv)
        else:
            impact = 0.0
        return (
            impact,
            self.half_spread_bp,
            self.commission_bp,
            impact + self.half_spread_bp + self.commission_bp,
        )

    def _apply_slippage(
        self, price: float, order_value: float, adv: float, direction: int
    ) -> tuple[float, float]:
        _, _, _, bp = self._slippage_components(order_value, adv)
        return price * (1.0 + direction * bp / 10000.0), bp

    # --------------------------------------------------------------- helpers
    def _prev_adv(self, px: pd.DataFrame, code: str, d) -> float:
        try:
            g = px.loc[code]
        except KeyError:
            return 0.0
        prev = g.loc[g.index < d]
        if prev.empty:
            return 0.0
        return float(prev.tail(self.adv_window)["turnover"].mean())

    def _get_future_row(self, px: pd.DataFrame, code: str, d, holding_days: int):
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

    def _latest_shortability_before(self, sh: pd.DataFrame, code: str, d) -> dict | None:
        try:
            g = sh.loc[code]
        except KeyError:
            return None
        prev = g.loc[g.index <= d]
        if prev.empty:
            return None
        return dict(prev.iloc[-1])

    def _entry_price(
        self,
        row: pd.Series,
        sig: dict,
        adv: float,
        side_sign: int,
    ) -> tuple[float | None, float]:
        order_type = str(sig.get("order_type", "MKT_OPEN")).upper()
        qty = int(sig.get("qty", 0))

        limit_price: float | None = None

        if order_type == "LIMIT":
            raw_limit = sig.get("limit_price")

            try:
                limit_price = float(raw_limit)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None, 0.0

            if not np.isfinite(limit_price) or limit_price <= 0:
                return None, 0.0

            if side_sign > 0:
                if not self._fills_limit_buy(float(row["low"]), limit_price):
                    return None, 0.0
                base = min(limit_price, float(pd.to_numeric(row["open"])))
            else:
                if not self._fills_limit_sell(float(row["high"]), limit_price):
                    return None, 0.0
                base = max(limit_price, float(pd.to_numeric(row["open"])))

        elif order_type == "MKT_CLOSE":
            base = float(pd.to_numeric(row["close"]))

        elif order_type == "MKT_OPEN":
            base = float(pd.to_numeric(row["open"]))

        else:
            return None, 0.0

        order_value = float(pd.to_numeric(base)) * float(qty)
        execution_price, slip_bp = self._apply_slippage(
            base,
            order_value,
            adv,
            direction=side_sign,
        )

        # 指値より不利な価格では約定させない。
        if limit_price is not None:
            if side_sign > 0 and execution_price > limit_price:
                return None, 0.0
            if side_sign < 0 and execution_price < limit_price:
                return None, 0.0

        return execution_price, slip_bp

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        shortability: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
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
            base = s.to_dict()
            code = s["code"]
            d = pd.to_datetime(s["date"])

            if s["side"] not in _VALID_SIDES:
                results.append({**base, "entry_date": str(d.date()), "status": "INVALID_SIDE"})
                continue

            try:
                qty = int(s.get("qty", 0))
            except (TypeError, ValueError):
                qty = 0
            if qty <= 0:
                results.append({**base, "entry_date": str(d.date()), "status": "INVALID_QTY"})
                continue

            if (code, d) not in px.index:
                results.append({**base, "entry_date": str(d.date()), "status": "NO_PRICE_DATA"})
                continue

            row = px.loc[(code, d)]
            adv = self._prev_adv(px, code, d)

            if self.require_liquidity_data and not self._is_valid_adv(adv):
                results.append(
                    {
                        **base,
                        "entry_date": str(d.date()),
                        "status": "NO_ENTRY_LIQUIDITY_DATA",
                    }
                )
                continue

            # FR-BT-05: 売り可否チェック
            if s["side"] == "SELL" and self.require_confirmed_shortability:
                snap = self._latest_shortability_before(sh, code, d) if sh is not None else None
                shortable = (
                    snap is not None
                    and int(snap["is_margin_lendable"]) == 1
                    and int(snap["short_restricted"]) == 0
                )
                if not shortable:
                    results.append(
                        {**base, "entry_date": str(d.date()), "status": "SKIP_NOT_SHORTABLE"}
                    )
                    continue

            side_sign = 1 if s["side"] == "BUY" else -1
            entry, slip_entry_bp = self._entry_price(row, s, adv, side_sign)
            if entry is None:
                results.append({**base, "entry_date": str(d.date()), "status": "NO_FILL"})
                continue

            try:
                holding_days = int(s.get("holding_days", 1))
            except (TypeError, ValueError):
                holding_days = 0

            if holding_days < 1:
                results.append(
                    {
                        **base,
                        "entry_date": str(d.date()),
                        "status": "INVALID_HOLDING_DAYS",
                    }
                )
                continue

            exit_row, exit_date = self._get_future_row(
                px,
                code,
                d,
                holding_days,
            )

            if exit_row is None or exit_date is None:
                results.append(
                    {
                        **base,
                        "entry_date": str(d.date()),
                        "status": "NO_EXIT_DATA",
                    }
                )
                continue

            exit_adv = self._prev_adv(px, code, exit_date)

            if self.require_liquidity_data and not self._is_valid_adv(exit_adv):
                results.append(
                    {
                        **base,
                        "entry_date": str(d.date()),
                        "exit_date": str(pd.Timestamp(exit_date).date()),
                        "status": "NO_EXIT_LIQUIDITY_DATA",
                    }
                )
                continue

            exit_base = float(exit_row["close"])
            exit_order_value = exit_base * qty

            exit_price, slip_exit_bp = self._apply_slippage(
                exit_base,
                exit_order_value,
                exit_adv,
                direction=-side_sign,
            )

            # 翌営業日決済はintradayではない。
            cal_days = max(
                (pd.Timestamp(exit_date) - pd.Timestamp(d)).days,
                0,
            )

            if self.zero_carry_for_intraday and cal_days == 0:
                carry_cost_per_share = 0.0
            else:
                carry_rate = self.daily_lend if s["side"] == "SELL" else self.daily_int
                carry_cost_per_share = entry * carry_rate * cal_days

            direction = side_sign
            gross_pnl = (exit_price - entry) * direction * qty
            carry_cost = carry_cost_per_share * qty
            pnl = gross_pnl - carry_cost

            results.append(
                {
                    **base,
                    "entry_date": str(pd.Timestamp(d).date()),
                    "exit_date": str(pd.Timestamp(exit_date).date()),
                    "entry": entry,
                    "exit": exit_price,
                    "entry_adv": adv,
                    "exit_adv": exit_adv,
                    "carry_cost_per_share": carry_cost_per_share,
                    "carry_cost": carry_cost,
                    "carry_days": cal_days,
                    "slippage_entry_bp": slip_entry_bp,
                    "slippage_exit_bp": slip_exit_bp,
                    "gross_pnl": gross_pnl,
                    "pnl": pnl,
                    "status": "FILLED",
                }
            )

        return pd.DataFrame(results)
