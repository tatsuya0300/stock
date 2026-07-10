"""シグナル生成モデル（FR-MODEL-01/02/04）。

SignalModel インターフェースを介してルールベース/ML を差し替え可能にする。
指数方向は予測せず、相対的な上位/下位を狙う設計。

look-ahead 回避:
  当日終値は as_of 時点で未確定のため使わない。
  前営業日までの価格のみ使用する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .calendar import previous_business_day


class SignalModel(ABC):
    """シグナル生成インターフェース。"""

    @abstractmethod
    def generate(self, prices: pd.DataFrame, as_of: str) -> pd.DataFrame:
        """returns: code, side('BUY'|'SELL'), score, limit_price(optional)"""
        raise NotImplementedError


class MeanReversionRule(SignalModel):
    """ルールベースの例（動作確認用ダミー）。収益性は担保しない。"""

    def __init__(self, lookback: int = 5, top_n: int = 5):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        self.lookback = lookback
        self.top_n = top_n

    def generate(self, prices: pd.DataFrame, as_of: str) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["code", "side", "score", "limit_price"])
        if prices is None or prices.empty:
            return empty

        as_of_d = pd.Timestamp(as_of).date()
        # 寄前想定: 当日終値は未確定のため使わない
        # 祝日・年末年始も calendar.previous_business_day で除外
        cutoff_d = previous_business_day(as_of_d)
        cutoff = cutoff_d.isoformat()

        df = prices[prices["date"] <= cutoff].copy()
        if df.empty:
            return empty
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"])

        # リターン計算は分割・配当調整済みの adj_close を使う。
        price_col = "adj_close" if "adj_close" in df.columns else "close"

        rets: dict[str, float] = {}
        for code, g in df.groupby("code"):
            g = g.tail(self.lookback + 1)
            if len(g) < self.lookback + 1:
                continue
            first = g[price_col].iloc[0]
            last = g[price_col].iloc[-1]
            if first is not None and first > 0 and np.isfinite(first) and np.isfinite(last):
                rets[str(code)] = (last / first) - 1.0

        if len(rets) < 2:
            return empty

        ser = pd.Series(rets, dtype=float).dropna().sort_values()

        # BUY/SELLが重複しない最大数
        n = min(
            self.top_n,
            len(ser) // 2,
        )

        if n < 1:
            return empty

        buys = ser.iloc[:n]
        sells = ser.iloc[-n:]

        # 防御的チェック
        overlap = set(buys.index) & set(sells.index)
        if overlap:
            raise RuntimeError(f"BUY/SELL overlap detected: {sorted(overlap)}")

        rows: list[dict] = []
        for code, r in buys.items():
            rows.append(
                {
                    "code": code,
                    "side": "BUY",
                    "score": float(-r),
                    "limit_price": np.nan,
                }
            )
        for code, r in sells.items():
            rows.append(
                {
                    "code": code,
                    "side": "SELL",
                    "score": float(r),
                    "limit_price": np.nan,
                }
            )

        return pd.DataFrame(rows, columns=["code", "side", "score", "limit_price"])
