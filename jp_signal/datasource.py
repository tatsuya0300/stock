"""価格データソース。

設計原則:
- 約定・売買金額には raw OHLC を使う。
- 特徴量・リターンには adjusted OHLC を使う。
- datasource 層で raw / adjusted を分離した統一スキーマに変換する。

注意:
- yfinance の end は exclusive。
- yfinance の turnover は close*volume の近似（本番は JQuants Va/TurnoverValue を使用）。
- J-Quants は V2 API（API Key 認証）を使用する。
  公式: https://jpx-jquants.com/en/spec/migration-v1-v2
        https://jpx-jquants.com/en/spec/eq-bars-daily
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any

import pandas as pd

from .data_quality import REQUIRED_PRICE_COLS, validate_prices
from .jquants_limits import resolve_jquants_sleep_sec
from .exceptions import (
    AuthenticationError,
    RateLimitError,
    RequestError,
    ResponseSchemaError,
)
from .universe import normalize_code

log = logging.getLogger(__name__)

OUTPUT_COLS = REQUIRED_PRICE_COLS

# プラン別レート制限は jquants_limits モジュールに集約


class PriceDataSource(ABC):
    """日足価格データ取得インターフェース。"""

    @abstractmethod
    def fetch_daily(self, codes: list[str], start: date, end: date) -> pd.DataFrame:
        """日足価格を統一スキーマで返す。

        returns columns:
          code, date,
          open, high, low, close,
          adj_open, adj_high, adj_low, adj_close,
          volume, turnover
        """
        raise NotImplementedError


def _requests_session_with_retry():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _empty_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLS)


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.columns = out.columns.get_level_values(0)
        return out
    return df


def _standardize_yfinance_frame(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """yfinance 戻り値を統一スキーマへ。

    yfinance は raw OHLC + Adj Close を返す。
    adj_open/high/low は adj_close/close 比率で近似（特徴量用途限定）。
    turnover は close*volume の近似であり、売買代金の真値ではない。
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

    close = pd.to_numeric(x["close"], errors="coerce")
    adj_close = pd.to_numeric(x["adj_close"], errors="coerce")
    ratio = adj_close / close
    ratio = ratio.replace([float("inf"), float("-inf")], pd.NA).fillna(1.0)

    x["adj_open"] = pd.to_numeric(x["open"], errors="coerce") * ratio
    x["adj_high"] = pd.to_numeric(x["high"], errors="coerce") * ratio
    x["adj_low"] = pd.to_numeric(x["low"], errors="coerce") * ratio
    x["code"] = normalize_code(code)
    # 近似: 真の売買代金ではない。本番インパクト算定は JQuants を使うこと。
    x["turnover"] = close * pd.to_numeric(x["volume"], errors="coerce")
    x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")

    return x[OUTPUT_COLS]


def _to_jquants_code(code: str) -> str:
    """内部4桁コードを J-Quants の発行体コードへ。

    公式は 4桁（普通株）または 5桁（例: 86970）を受け付ける。
    4桁指定時は普通株のみ取得される。
    参考: https://jpx-jquants.com/en/spec/eq-bars-daily
    """
    c = normalize_code(code)
    if c.isdigit() and len(c) == 4:
        return c
    return c


def _from_jquants_code(code: str) -> str:
    """J-Quants の Code を内部4桁へ正規化。

    例: '86970' -> '8697', '7203' -> '7203'
    """
    c = str(code).strip()
    if c.isdigit() and len(c) == 5 and c.endswith("0"):
        return c[:4]
    return normalize_code(c)


def _standardize_jquants_v2_frame(df: pd.DataFrame) -> pd.DataFrame:
    """J-Quants V2 /v2/equities/bars/daily を統一スキーマへ。

    V2 列名:
      Date, Code, O, H, L, C, Vo, Va, AdjO, AdjH, AdjL, AdjC, ...
    """
    if df is None or df.empty:
        return _empty_prices()

    x = df.copy()
    rename_map = {
        "Date": "date",
        "Code": "code",
        # V2 短縮形
        "O": "open",
        "H": "high",
        "L": "low",
        "C": "close",
        "AdjO": "adj_open",
        "AdjH": "adj_high",
        "AdjL": "adj_low",
        "AdjC": "adj_close",
        "Vo": "volume",
        "Va": "turnover",
        # V1 形式（互換）
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
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
            "公式仕様を確認してください: "
            "https://jpx-jquants.com/en/spec/eq-bars-daily"
        )

    x["code"] = x["code"].astype(str).map(_from_jquants_code)
    x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")

    for c in [
        "open",
        "high",
        "low",
        "close",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "volume",
        "turnover",
    ]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    return x[OUTPUT_COLS]


class YFinanceSource(PriceDataSource):
    """プロトタイプ用 yfinance datasource。

    注意:
      turnover は close*volume の近似。
      マーケットインパクト・ADV 上限の本番算定には使わないこと。
    """

    # 近似フラグ（呼び出し側が本番判定に使える）
    TURNOVER_IS_APPROXIMATE = True

    def __init__(self, *, strict_data_quality: bool = False, chunk_size: int = 50):
        self.strict_data_quality = strict_data_quality
        self.chunk_size = max(1, int(chunk_size))
        log.warning("YFinanceSource: turnover=close*volume の近似。本番は JQuantsSource を使用。")

    def fetch_daily(self, codes: list[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        if not codes:
            return _empty_prices()

        # end exclusive 対策
        end_exclusive = end + timedelta(days=1)

        norm_codes = [normalize_code(c) for c in codes]
        symbols = [f"{c}.T" for c in norm_codes]
        code_map = {f"{c}.T": c for c in norm_codes}

        frames: list[pd.DataFrame] = []
        failed: list[str] = []

        for i in range(0, len(symbols), self.chunk_size):
            chunk = symbols[i : i + self.chunk_size]
            try:
                raw = yf.download(
                    tickers=chunk,
                    start=start.isoformat(),
                    end=end_exclusive.isoformat(),
                    auto_adjust=False,
                    repair=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    timeout=60,
                )
            except Exception as exc:
                log.warning("yfinance chunk failed: n=%d error=%s", len(chunk), exc)
                failed.extend(chunk)
                continue

            if raw is None or raw.empty:
                failed.extend(chunk)
                continue

            # 単一銘柄は MultiIndex にならないことがある
            if len(chunk) == 1:
                sym = chunk[0]
                try:
                    std = _standardize_yfinance_frame(raw, code_map[sym])
                    if std.empty:
                        failed.append(sym)
                    else:
                        frames.append(std)
                except Exception as exc:
                    log.warning("yfinance standardize failed: %s error=%s", sym, exc)
                    failed.append(sym)
                continue

            # 複数銘柄
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
            else:
                log.warning("unexpected yfinance columns format for chunk size=%d", len(chunk))
                failed.extend(chunk)
                continue

            for sym in chunk:
                if sym not in level0:
                    failed.append(sym)
                    continue
                try:
                    part = raw[sym].dropna(how="all")
                    std = _standardize_yfinance_frame(part, code_map[sym])
                    if std.empty:
                        failed.append(sym)
                    else:
                        frames.append(std)
                except Exception as exc:
                    log.warning("yfinance standardize failed: %s error=%s", sym, exc)
                    failed.append(sym)

        if failed:
            log.warning(
                "yfinance failed/empty for %d symbols (show 10): %s",
                len(failed),
                failed[:10],
            )

        if not frames:
            return _empty_prices()

        out = pd.concat(frames, ignore_index=True)
        # inclusive end を保証
        out = out[out["date"] <= end.isoformat()]
        return validate_prices(out, strict=self.strict_data_quality)


class JQuantsSource(PriceDataSource):
    """J-Quants V2 datasource。

    V2 認証: API Key (x-api-key header)。
    株価 endpoint: /v2/equities/bars/daily。
    応答キー: data（旧 daily_quotes ではない）。

    参考:
      https://jpx-jquants.com/en/spec/migration-v1-v2
      https://jpx-jquants.com/en/spec/eq-bars-daily
    """

    BASE = "https://api.jquants.com/v2"

    def __init__(
        self,
        api_key: str,
        *,
        strict_data_quality: bool = True,
        sleep_sec: float | None = None,
        plan: str = "free",
    ):
        if not api_key:
            raise ValueError("JQUANTS_API_KEY is required for JQuantsSource (V2).")

        normalized_plan, resolved_sleep_sec = resolve_jquants_sleep_sec(
            plan=plan,
            sleep_sec=sleep_sec,
        )

        self.api_key = api_key
        self.plan = normalized_plan
        self.strict_data_quality = strict_data_quality
        self.sleep_sec = resolved_sleep_sec
        self._last_request_ts = 0.0
        self.max_retries_on_429 = 3
        self._session = _requests_session_with_retry()

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    def _throttle(self) -> None:
        """プラン別最小間隔を保証。"""
        elapsed = time.monotonic() - self._last_request_ts
        wait = self.sleep_sec - elapsed
        if wait > 0:
            time.sleep(wait)

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries_on_429 + 1):
            self._throttle()
            r = self._session.get(
                f"{self.BASE}{path}",
                params=params,
                headers=self._headers(),
                timeout=30,
            )
            self._last_request_ts = time.monotonic()

            if r.status_code == 401:
                raise PermissionError(
                    "J-Quants API Key が無効です。ダッシュボードで再発行してください。"
                )
            if r.status_code == 429:
                # Retry-After があれば尊重、無ければ指数バックオフ
                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        backoff = float(retry_after)
                    except ValueError:
                        backoff = self.sleep_sec * (2**attempt)
                else:
                    backoff = self.sleep_sec * (2**attempt)
                backoff = min(max(backoff, self.sleep_sec), 120.0)
                log.warning(
                    "J-Quants rate limited (429). attempt=%d/%d sleep=%.1fs params=%s",
                    attempt + 1,
                    self.max_retries_on_429 + 1,
                    backoff,
                    params,
                )
                if attempt >= self.max_retries_on_429:
                    r.raise_for_status()
                time.sleep(backoff)
                last_exc = RuntimeError("J-Quants 429 rate limit")
                continue

            r.raise_for_status()
            js = r.json()
            if not isinstance(js, dict):
                raise TypeError("J-Quants response must be a JSON object.")
            return js

        raise RuntimeError(f"J-Quants request failed after retries: {last_exc}")

    def _fetch_code(self, code: str, start: date, end: date) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        pagination_key: str | None = None
        jq_code = _to_jquants_code(code)

        while True:
            params: dict[str, Any] = {
                "code": jq_code,
                "from": start.isoformat(),
                "to": end.isoformat(),
            }
            if pagination_key:
                params["pagination_key"] = pagination_key

            js = self._get_json("/equities/bars/daily", params)
            batch = js.get("data", [])
            if batch is None:
                batch = []
            if not isinstance(batch, list):
                raise TypeError("J-Quants data must be a list.")

            rows.extend(batch)

            pagination_key = js.get("pagination_key")
            if not pagination_key:
                break

        return pd.DataFrame(rows)

    def fetch_daily(self, codes: list[str], start: date, end: date) -> pd.DataFrame:
        if not codes:
            return _empty_prices()

        frames: list[pd.DataFrame] = []
        failed: list[str] = []

        for raw_code in codes:
            c = normalize_code(raw_code)
            try:
                df = self._fetch_code(c, start, end)
                std = _standardize_jquants_v2_frame(df)
                if std.empty:
                    failed.append(c)
                else:
                    frames.append(std)
            except Exception as exc:
                log.warning("J-Quants fetch failed: code=%s error=%s", c, exc)
                failed.append(c)
                continue

        if failed:
            log.warning(
                "J-Quants failed/empty for %d codes (show 10): %s",
                len(failed),
                failed[:10],
            )

        if not frames:
            return _empty_prices()

        out = pd.concat(frames, ignore_index=True)
        out = out[out["date"] <= end.isoformat()]
        return validate_prices(out, strict=self.strict_data_quality)
