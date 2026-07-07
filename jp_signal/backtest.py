"""バックテストエンジン（FR-BT-01〜05）。改訂版。

主な修正点（改訂前からの差分）:
  1. エグジット（決済）にも反対売買のマーケットインパクトを適用（往復コスト計上）。
  2. _get_future_row が (row, exit_date) を返し、exit 日の前日 ADV を分母に使用。
  3. スプレッド/手数料フックを追加（デフォルト0だが将来拡張のため引数化）。

約定・コストモデル:
  - FR-BT-01: 買い指値 P は当日安値 < P で約定（同値未約定は保守仮定。感度分析を別途行うこと）。
              売り指値 P は当日高値 > P で約定（同値未約定）。
  - FR-BT-02: 売買手数料はデフォルト0（commission_bp で変更可）。
  - FR-BT-03: 信用金利/貸株料 年率2%（日次 = rate/365）、オーバーナイト保有日数で計上。
  - FR-BT-04: マーケットインパクト sqrt則 impact_bp = k * sqrt(order_value / adv)。
              k はデータソース依存の較正値。yfinance 近似 turnover と JQuants 実測は別較正。
  - FR-BT-05: 売り戦略は shortability(is_margin_lendable=1 かつ short_restricted=0) の銘柄のみ。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Backtester:
    """日次シグナルを受け取り、往復の約定・コストを反映して PnL を算出する。"""

    def __init__(
        self,
        impact_k_bp: float = 30.0,
        annual_interest_rate: float = 0.02,
        annual_lending_rate: float = 0.02,
        commission_bp: float = 0.0,
        half_spread_bp: float = 0.0,
    ):
        self.k = impact_k_bp
        self.daily_int = annual_interest_rate / 365.0
        self.daily_lend = annual_lending_rate / 365.0
        self.commission_bp = commission_bp
        self.half_spread_bp = half_spread_bp

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
        """1回の約定にかかる不利方向スリッページ(bp) = インパクト + ハーフスプレッド + 手数料。"""
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
    def _prev_turnover(self, px: pd.DataFrame, code: str, d) -> float:
        """d より前の直近営業日の売買代金（ADV近似）。存在しなければ 0。"""
        try:
            g = px.loc[code]
        except KeyError:
            return 0.0
        prev = g.loc[g.index < d]
        if prev.empty:
            return 0.0
        return float(prev.iloc[-1]["turnover"])

    def _get_future_row(self, px: pd.DataFrame, code: str, d, holding_days: int):
        """d から holding_days 営業日後の (行, その日付) を返す。末尾超えは (None, None)。"""
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
                base = min(lp, row["open"])  # ギャップダウン考慮
            else:  # SELL
                if not self._fills_limit_sell(row["high"], lp):
                    return None
                base = max(lp, row["open"])  # ギャップアップ考慮
        elif order_type == "MKT_CLOSE":
            base = row["close"]
        else:  # MKT_OPEN（デフォルト）
            base = row["open"]

        order_value = base * qty
        return self._apply_slippage(base, order_value, adv, direction=side_sign)

    # ------------------------------------------------------------------- run
    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        shortability: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """バックテストを実行して 1 シグナル 1 行の結果 DataFrame を返す。

        signals: date, code, side, qty, order_type('MKT_OPEN'|'MKT_CLOSE'|'LIMIT'),
                 limit_price(nullable), holding_days
        prices : code,date,open,high,low,close,volume,turnover
        """
        if signals is None or signals.empty or prices is None or prices.empty:
            return pd.DataFrame()

        px = prices.copy()
        px["date"] = pd.to_datetime(px["date"])
        px = px.set_index(["code", "date"]).sort_index()

        if shortability is not None and not shortability.empty:
            sh = shortability.copy()
            sh["date"] = pd.to_datetime(sh["date"])
            sh = sh.set_index(["code", "date"])
        else:
            sh = None

        results = []
        for _, s in signals.iterrows():
            code = s["code"]
            d = pd.to_datetime(s["date"])
            if (code, d) not in px.index:
                results.append({**s.to_dict(), "status": "NO_PRICE_DATA"})
                continue
            row = px.loc[(code, d)]
            adv = self._prev_turnover(px, code, d)

            # 売り可否チェック（FR-BT-05）。データ欠損時は保守側=売り不可。
            if s["side"] == "SELL":
                shortable = (
                    sh is not None
                    and (code, d) in sh.index
                    and int(sh.loc[(code, d), "is_margin_lendable"]) == 1
                    and int(sh.loc[(code, d), "short_restricted"]) == 0
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

            qty = s["qty"]
            # 決済は反対売買。exit 日の前日 ADV を分母に使う。
            exit_base = float(exit_row["close"])
            exit_adv = self._prev_turnover(px, code, exit_date)
            exit_price = self._apply_slippage(
                exit_base, exit_base * qty, exit_adv, direction=-side_sign
            )

            # 金利/貸株（1株あたり、エントリー価格ベース）
            carry_rate = self.daily_lend if s["side"] == "SELL" else self.daily_int
            carry_cost = entry * carry_rate * holding_days

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
