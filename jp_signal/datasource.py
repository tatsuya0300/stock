"""価格データソース（FR-DATA-01/03/05）。改訂版。

PriceDataSource インターフェースを介して yfinance（プロトタイプ）と
JQuants（本番）を差し替え可能にする。

改訂点:
  - yfinance: auto_adjust=False で Close と Adj Close を両方取得し、adj_close を
    別列として保持する。リターン計算は adj_close、約定金額は生 close/open を使う
    （分割・配当調整のダブルカウント/取りこぼしバグ対策）。
  - JQuants: pagination_key ループ、idToken の有効期限管理、リトライ/バックオフを追加。

一次情報:
  - JQuants API 仕様: https://jpx.gitbook.io/j-quants-ja
  - yfinance: https://github.com/ranaroussi/yfinance

注記（ハルシネーション回避）:
  JQuants の列名（AdjustmentClose 等）・pagination_key というキー名・idToken の
  有効期限は記憶ベースの推定を含む。本番接続前に必ず公式ドキュメントで裏取りすること。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta

import pandas as pd

OUTPUT_COLS = [
    "code", "date", "open", "high", "low", "close",
    "adj_close", "volume", "turnover",
]


class PriceDataSource(ABC):
    """日足価格の取得インターフェース。"""

    @abstractmethod
    def fetch_daily(self, codes: list[str], start: date, end: date) -> pd.DataFrame:
        """returns columns: code,date,open,high,low,close,adj_close,volume,turnover"""
        raise NotImplementedError


def _requests_session_with_retry():
    """指数バックオフ付きの requests.Session を返す。"""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class YFinanceSource(PriceDataSource):
    """プロトタイプ用。銘柄コードは '7203.T' 形式に変換。

    close は生の終値、adj_close は分割・配当調整後終値。
    リターン計算は adj_close を、約定金額は close/open を使うこと。
    turnover は close*volume で近似（本番では JQuants の TurnoverValue を使用）。
    """

    def fetch_daily(self, codes: list[str], start, end) -> pd.DataFrame:
        import yfinance as yf

        frames: list[pd.DataFrame] = []
        for c in codes:
            sym = f"{c}.T"
            try:
                # auto_adjust=False で Close と Adj Close の両方を取得
                df = yf.download(
                    sym, start=start, end=end, auto_adjust=False,
                    progress=False, threads=False,
                )
            except Exception as exc:  # 当該銘柄のみスキップ（他銘柄は継続）
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
                    "Low": "low", "Close": "close", "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            if "adj_close" not in df.columns:
                # 一部シンボルで Adj Close が無い場合は close で代替（警告）
                print(f"[YFinanceSource] {sym}: Adj Close 欠損, close で代替")
                df["adj_close"] = df["close"]

            df["code"] = c
            df["turnover"] = df["close"] * df["volume"]  # 近似値
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            frames.append(df[OUTPUT_COLS])

        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=OUTPUT_COLS)
        )


class JQuantsSource(PriceDataSource):
    """本番用。pagination_key ループとトークン有効期限管理を実装。

    正確なエンドポイント・レート制限・列名は公式ドキュメントで確認すること:
    https://jpx.gitbook.io/j-quants-ja
    idToken の有効期限は公式仕様に従う（本実装では保守的に短めに設定）。
    """

    BASE = "https://api.jquants.com/v1"
    # 公式仕様上の idToken 有効期限に基づき、余裕を持って失効扱いにするマージン
    _ID_TOKEN_TTL = timedelta(hours=20)

    def __init__(self, refresh_token: str):
        self.refresh_token = refresh_token
        self._id_token: str | None = None
        self._id_token_at: datetime | None = None
        self._session = _requests_session_with_retry()

    def _auth(self) -> str:
        now = datetime.utcnow()
        if (
            self._id_token
            and self._id_token_at
            and now - self._id_token_at < self._ID_TOKEN_TTL
        ):
            return self._id_token
        r = self._session.post(
            f"{self.BASE}/token/auth_refresh",
            params={"refreshtoken": self.refresh_token},
            timeout=30,
        )
        r.raise_for_status()
        self._id_token = r.json()["idToken"]
        self._id_token_at = now
        return self._id_token

    def _fetch_code(self, code, start, end, headers) -> pd.DataFrame:
        """1銘柄分を pagination_key を辿って全件取得する。"""
        rows: list[dict] = []
        pagination_key: str | None = None
        while True:
            params = {"code": code, "from": str(start), "to": str(end)}
            if pagination_key:
                params["pagination_key"] = pagination_key
            r = self._session.get(
                f"{self.BASE}/prices/daily_quotes",
                params=params, headers=headers, timeout=30,
            )
            r.raise_for_status()
            js = r.json()
            rows.extend(js.get("daily_quotes", []))
            pagination_key = js.get("pagination_key")
            if not pagination_key:
                break
            time.sleep(0.2)  # レート制限緩和
        return pd.DataFrame(rows)

    def fetch_daily(self, codes, start, end) -> pd.DataFrame:
        token = self._auth()
        headers = {"Authorization": f"Bearer {token}"}
        frames: list[pd.DataFrame] = []
        for c in codes:
            df = self._fetch_code(c, start, end, headers)
            if df.empty:
                continue
            df = df.rename(
                columns={
                    "Date": "date", "Code": "code",
                    "AdjustmentOpen": "open", "AdjustmentHigh": "high",
                    "AdjustmentLow": "low", "AdjustmentClose": "adj_close",
                    "Close": "close", "Open": "open_raw",
                    "Volume": "volume", "TurnoverValue": "turnover",
                }
            )
            # 注記: JQuants の列名は仕様変更されうる。上記マッピングは要確認。
            # Adjustment* は調整後、無印は生値。分割時のリターン計算は adj_close を使う。
            missing = [c2 for c2 in OUTPUT_COLS if c2 not in df.columns]
            if missing:
                raise KeyError(
                    f"JQuants 応答に必要列が不足: {missing}. "
                    "公式ドキュメント(https://jpx.gitbook.io/j-quants-ja)で列名を確認してください。"
                )
            frames.append(df[OUTPUT_COLS])

        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=OUTPUT_COLS)
        )
