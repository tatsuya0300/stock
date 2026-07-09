"""価格データソース。

設計原則:
- 約定・売買金額には raw open/high/low/close を使う。
- 特徴量・リターンには adjusted open/high/low/close を使う。
- datasource 層で raw と adjusted を分離した統一スキーマに変換する。

注意:
- yfinance の end は exclusive。
- J-Quants の列名・pagination key 名・エンドポイントは公式仕様で確認が必要。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from .data_quality import REQUIRED_PRICE_COLS, validate_prices
from .universe import normalize_code

log = logging.getLogger(__name__)

OUTPUT_COLS = REQUIRED_PRICE_COLS


class PriceDataSource(ABC):
    """日足価格データ取得インターフェース。"""

    @abstractmethod
    def fetch_daily(self, codes: list[str], start: date, end: date) -> pd.DataFrame:
        """日足価格を統一スキーマで返す。

        returns:
            columns:
            code, date,
            open, high, low, close,
            adj_open, adj_high, adj_low, adj_close,
            volume, turnover
        """
        raise NotImplementedError


def _requests_session_with_retry():
    """指数バックオフ付き requests.Session を返す。"""
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


def _empty_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLS)


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance が MultiIndex 列を返す場合に平坦化する。"""
    if isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.columns = out.columns.get_level_values(0)
        return out
    return df


def _standardize_yfinance_frame(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """yfinance の戻り値を統一スキーマへ変換する。

    yfinance は raw OHLC と Adj Close は返すが、adjusted OHLC は直接返さない。
    そのため adj_open/high/low は adj_close / close の比率で近似する。
    ただし、約定には raw OHLC を使うので、この近似値は特徴量用途に限定する。
    """
    if df is None or df.empty:
        return _empty_prices()

    x = _flatten_yfinance_columns(df).copy()

    x = x.reset_index().rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    required_raw = ["date", "open", "high", "low", "close", "volume"]
    missing_raw = [c for c in required_raw if c not in x.columns]
    if missing_raw:
        raise KeyError(f"yfinance response missing columns: {missing_raw}")

    if "adj_close" not in x.columns:
        log.warning("%s: Adj Close missing; fallback adj_close=close", code)
        x["adj_close"] = x["close"]

    # close が0または欠損の場合の調整比率は 1.0 にフォールバック。
    close = pd.to_numeric(x["close"], errors="coerce")
    adj_close = pd.to_numeric(x["adj_close"], errors="coerce")
    ratio = adj_close / close
    ratio = ratio.replace([float("inf"), float("-inf")], pd.NA).fillna(1.0)

    x["adj_open"] = pd.to_numeric(x["open"], errors="coerce") * ratio
    x["adj_high"] = pd.to_numeric(x["high"], errors="coerce") * ratio
    x["adj_low"] = pd.to_numeric(x["low"], errors="coerce") * ratio

    x["code"] = normalize_code(code)
    x["turnover"] = pd.to_numeric(x["close"], errors="coerce") * pd.to_numeric(
        x["volume"], errors="coerce"
    )
    x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")

    return x[OUTPUT_COLS]


def _standardize_jquants_frame(df: pd.DataFrame) -> pd.DataFrame:
    """J-Quants の daily_quotes 応答を統一スキーマへ変換する。

    注意:
    以下の列名は J-Quants API の仕様に依存する。
    本番運用前に公式仕様で必ず確認すること。
    """
    if df is None or df.empty:
        return _empty_prices()

    x = df.copy()

    rename_map = {
        "Date": "date",
        "Code": "code",
        # raw OHLC: 約定・売買金額用
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        # adjusted OHLC: 特徴量・リターン用
        "AdjustmentOpen": "adj_open",
        "AdjustmentHigh": "adj_high",
        "AdjustmentLow": "adj_low",
        "AdjustmentClose": "adj_close",
        "Volume": "volume",
        "TurnoverValue": "turnover",
    }

    x = x.rename(columns=rename_map)

    missing = [col for col in OUTPUT_COLS if col not in x.columns]
    if missing:
        raise KeyError(
            f"J-Quants response missing columns: {missing}. "
            "J-Quants公式仕様で列名を確認してください。"
        )

    x["code"] = x["code"].astype(str).map(normalize_code)
    x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")

    return x[OUTPUT_COLS]


class YFinanceSource(PriceDataSource):
    """プロトタイプ用 yfinance datasource。

    yfinance.download の end は exclusive。
    例:
        start=2024-01-01, end=2024-01-06
        -> 2024-01-05 まで取得される。
    """

    def __init__(self, *, strict_data_quality: bool = False):
        self.strict_data_quality = strict_data_quality

    def fetch_daily(self, codes: list[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        frames: list[pd.DataFrame] = []

        for raw_code in codes:
            c = normalize_code(raw_code)
            sym = f"{c}.T"

            try:
                df = yf.download(
                    sym,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    repair=True,
                    progress=False,
                    threads=False,
                    timeout=30,
                )
                std = _standardize_yfinance_frame(df, c)
                if not std.empty:
                    frames.append(std)

            except Exception as exc:
                log.warning("yfinance fetch failed: symbol=%s error=%s", sym, exc)
                continue

        if not frames:
            return _empty_prices()

        out = pd.concat(frames, ignore_index=True)
        return validate_prices(out, strict=self.strict_data_quality)


class JQuantsSource(PriceDataSource):
    """J-Quants datasource。

    注意:
    - endpoint
    - response key
    - pagination key
    - idToken TTL
    - column names

    これらは J-Quants 公式仕様に依存する。
    本番運用前に必ず確認すること。
    """

    BASE = "https://api.jquants.com/v1"
    _ID_TOKEN_TTL = timedelta(hours=20)

    def __init__(self, refresh_token: str, *, strict_data_quality: bool = True):
        if not refresh_token:
            raise ValueError("refresh_token is required for JQuantsSource.")
        self.refresh_token = refresh_token
        self.strict_data_quality = strict_data_quality
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
            f"{self.BASE}/auth/refresh_token",
            params={"refreshtoken": self.refresh_token},
            timeout=30,
        )
        r.raise_for_status()
        js = r.json()
        self._id_token = str(js["idToken"])
        self._id_token_at = now
        return self._id_token

    def _fetch_code(
        self, code: str, start: date, end: date, headers: dict
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        pagination_key: str | None = None

        while True:
            params: dict[str, Any] = {
                "code": code,
                "from": str(start),
                "to": str(end),
            }
            if pagination_key:
                params["pagination_key"] = pagination_key

            r = self._session.get(
                f"{self.BASE}/prices/daily_quotes",
                params=params,
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()

            js = r.json()
            batch = js.get("daily_quotes", [])
            if not isinstance(batch, list):
                raise TypeError("J-Quants daily_quotes must be a list.")

            rows.extend(batch)

            pagination_key = js.get("pagination_key")
            if not pagination_key:
                break

            time.sleep(0.2)

        return pd.DataFrame(rows)

    def fetch_daily(self, codes: list[str], start: date, end: date) -> pd.DataFrame:
        token = self._auth()
        headers = {"Authorization": f"Bearer {token}"}

        frames: list[pd.DataFrame] = []

        for raw_code in codes:
            c = normalize_code(raw_code)
            try:
                df = self._fetch_code(c, start, end, headers)
                std = _standardize_jquants_frame(df)
                if not std.empty:
                    frames.append(std)
            except Exception as exc:
                log.warning("J-Quants fetch failed: code=%s error=%s", c, exc)
                continue

        if not frames:
            return _empty_prices()

        out = pd.concat(frames, ignore_index=True)
        return validate_prices(out, strict=self.strict_data_quality)
