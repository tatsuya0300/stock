"""価格データソース（FR-DATA-01/03/05）。

PriceDataSource インターフェースを介して yfinance（プロトタイプ）と
JQuants（本番）を差し替え可能にする。
一次情報:
  - JQuants API 仕様: https://jpx.gitbook.io/j-quants-ja
  - yfinance: https://github.com/ranaroussi/yfinance
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

OUTPUT_COLS = ["code", "date", "open", "high", "low", "close", "volume", "turnover"]


class PriceDataSource(ABC):
    """日足価格の取得インターフェース。"""

    @abstractmethod
    def fetch_daily(self, codes: list[str], start: date, end: date) -> pd.DataFrame:
        """returns columns: code,date,open,high,low,close,volume,turnover"""
        raise NotImplementedError


class YFinanceSource(PriceDataSource):
    """プロトタイプ用。銘柄コードは '7203.T' 形式に変換。

    turnover は close*volume で近似する（本番では JQuants の TurnoverValue を使用）。
    """

    def fetch_daily(self, codes: list[str], start, end) -> pd.DataFrame:
        import yfinance as yf

        frames: list[pd.DataFrame] = []
        for c in codes:
            sym = f"{c}.T"
            try:
                df = yf.download(
                    sym, start=start, end=end, auto_adjust=False,
                    progress=False, threads=False,
                )
            except Exception as exc:  # 取得失敗は当該銘柄のみスキップ（他銘柄は継続）
                print(f"[YFinanceSource] {sym} fetch failed: {exc}")
                continue
            if df is None or df.empty:
                continue

            # yfinance は MultiIndex 列を返すことがあるためフラット化
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index().rename(
                columns={
                    "Date": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                }
            )
            df["code"] = c
            df["turnover"] = df["close"] * df["volume"]  # 近似値
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            frames.append(df[OUTPUT_COLS])

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLS)


class JQuantsSource(PriceDataSource):
    """本番用スケルトン。

    正確なエンドポイント・レート制限・認証フローは公式ドキュメントで確認すること:
    https://jpx.gitbook.io/j-quants-ja
      - /token/auth_refresh でIDトークン取得
      - /prices/daily_quotes で日足取得（code, date 指定）
    列名（Date/Open/High/Low/Close/Volume/TurnoverValue）は要マッピング。
    """

    BASE = "https://api.jquants.com/v1"

    def __init__(self, refresh_token: str):
        self.refresh_token = refresh_token
        self._id_token: str | None = None

    def _auth(self) -> str:
        import requests

        if self._id_token:
            return self._id_token
        r = requests.post(
            f"{self.BASE}/token/auth_refresh",
            params={"refreshtoken": self.refresh_token},
            timeout=30,
        )
        r.raise_for_status()
        self._id_token = r.json()["idToken"]
        return self._id_token

    def fetch_daily(self, codes, start, end) -> pd.DataFrame:
        import requests

        token = self._auth()
        headers = {"Authorization": f"Bearer {token}"}
        frames: list[pd.DataFrame] = []
        for c in codes:
            params = {"code": c, "from": str(start), "to": str(end)}
            r = requests.get(
                f"{self.BASE}/prices/daily_quotes",
                params=params, headers=headers, timeout=30,
            )
            r.raise_for_status()
            js = r.json().get("daily_quotes", [])
            if not js:
                continue
            df = pd.DataFrame(js)
            # 列名は JQuants ドキュメントに準拠して要マッピング
            df = df.rename(
                columns={
                    "Date": "date", "Code": "code",
                    "Open": "open", "High": "high", "Low": "low", "Close": "close",
                    "Volume": "volume", "TurnoverValue": "turnover",
                }
            )
            # 必要列が欠ける場合に備えて存在チェック
            missing = [c2 for c2 in OUTPUT_COLS if c2 not in df.columns]
            if missing:
                raise KeyError(
                    f"JQuants 応答に必要列が不足: {missing}. 公式ドキュメントで列名を確認してください。"
                )
            frames.append(df[OUTPUT_COLS])

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLS)
