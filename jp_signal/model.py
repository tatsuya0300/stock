"""シグナル生成モデル（FR-MODEL-01/02/04）。

SignalModel インターフェースを介してルールベース/ML を差し替え可能にする。
指数方向は予測せず、相対的な上位/下位を狙う設計。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SignalModel(ABC):
    """シグナル生成インターフェース。"""

    @abstractmethod
    def generate(self, prices: pd.DataFrame, as_of: str) -> pd.DataFrame:
        """returns: code, side('BUY'|'SELL'), score, limit_price(optional)"""
        raise NotImplementedError


class MeanReversionRule(SignalModel):
    """ルールベースの例（動作確認用ダミー）。

    - lookback 日リターンの下位 top_n 銘柄 → BUY 候補
    - lookback 日リターンの上位 top_n 銘柄 → SELL 候補
    FR-MODEL-01 の「相対上位/下位」を満たすが収益性は担保しない。
    実運用前に必ずバックテストで検証すること。
    """

    def __init__(self, lookback: int = 5, top_n: int = 5):
        self.lookback = lookback
        self.top_n = top_n

    def generate(self, prices: pd.DataFrame, as_of: str) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["code", "side", "score", "limit_price"])
        if prices is None or prices.empty:
            return empty

        df = prices[prices["date"] <= as_of].copy()
        if df.empty:
            return empty
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"])

        # 各銘柄について lookback 日リターンを算出
        rets: dict[str, float] = {}
        for code, g in df.groupby("code"):
            g = g.tail(self.lookback + 1)
            if len(g) < self.lookback + 1:
                continue
            first = g["close"].iloc[0]
            last = g["close"].iloc[-1]
            if first and first > 0:
                rets[code] = (last / first) - 1.0

        if not rets:
            return empty

        ser = pd.Series(rets).sort_values()
        n = min(self.top_n, len(ser) // 2 if len(ser) >= 2 else len(ser))
        if n == 0:
            n = min(self.top_n, len(ser))

        buys = ser.head(n)   # 最も下落 → 買い
        sells = ser.tail(n)  # 最も上昇 → 売り

        rows = []
        for code, r in buys.items():
            rows.append({"code": code, "side": "BUY", "score": float(-r), "limit_price": np.nan})
        for code, r in sells.items():
            rows.append({"code": code, "side": "SELL", "score": float(r), "limit_price": np.nan})

        return pd.DataFrame(rows, columns=["code", "side", "score", "limit_price"])
