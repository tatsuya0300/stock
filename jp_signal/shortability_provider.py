"""空売り可能データソース抽象化＋CSV実装。

PR-2: ShortabilityDataSource を導入し、shortability 情報の
取得元を抽象化する。CsvShortabilityDataSource が CSV ファイルから
読み込む最小実装を提供する。

PR-4: ShortabilityDataProvider (PIT形式) と CsvShortabilityProvider を追加。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .shortability_pit import (
    normalize_shortability_observations,
)

log = logging.getLogger(__name__)


class ShortabilityDataSource(ABC):
    """空売り可否情報のデータソース抽象化。"""

    @abstractmethod
    def fetch(
        self,
        codes: list[str],
        decision_at: pd.Timestamp,
    ) -> pd.DataFrame:
        """指定銘柄群の空売り可否を返す。

        Returns:
            columns: code, shortable (bool), updated_at (str|None)
        """
        raise NotImplementedError


class CsvShortabilityDataSource(ShortabilityDataSource):
    """CSV ファイルから shortability を読み込む実装。"""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"shortability CSV not found: {self._path}")

    def fetch(
        self,
        codes: list[str],
        decision_at: pd.Timestamp,
    ) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame(columns=["code", "shortable", "updated_at"])

        raw = pd.read_csv(self._path, dtype=str)
        if raw.empty:
            return pd.DataFrame(columns=["code", "shortable", "updated_at"])

        raw["code"] = raw["code"].astype(str).str.strip()
        raw["shortable"] = raw.get("shortable", "false").astype(bool)
        raw["updated_at"] = raw.get("updated_at", None)

        mask = raw["code"].isin([str(c) for c in codes])
        result = raw[mask].copy()

        return result.reset_index(drop=True)


def refresh_shortability(
    ds: ShortabilityDataSource,
    codes: list[str],
    decision_at: pd.Timestamp,
) -> pd.DataFrame:
    """shortability を取得しバリデーションする簡便関数。"""
    result = ds.fetch(codes, decision_at)
    if result.empty:
        log.warning("shortability: 0 rows returned for %d codes", len(codes))
    return result


class ShortabilityDataProvider(ABC):
    """PIT形式の売建可否データProvider。

    外部データの取得形式を内部PIT形式から分離する。
    """

    @abstractmethod
    def fetch(
        self,
        *,
        fetched_at: pd.Timestamp,
    ) -> pd.DataFrame:
        """PIT shortability observationsを返す。"""
        raise NotImplementedError


class CsvShortabilityProvider(
    ShortabilityDataProvider
):
    """正式に取得したCSVをPIT形式へ変換する。

    必須列:
      code
      effective_at
      short_type
      is_shortable
      short_restricted

    systemの場合の必須列:
      is_margin_lendable

    任意列:
      source
      fetched_at
      stock_loan_fee_annual
    """

    def __init__(
        self,
        path: str | Path,
        *,
        default_source: str = "manual_csv",
    ):
        self.path = Path(path)
        self.default_source = (
            default_source
        )

    def fetch(
        self,
        *,
        fetched_at: pd.Timestamp,
    ) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"shortability CSV not found: "
                f"{self.path}"
            )

        frame = pd.read_csv(
            self.path,
            dtype={"code": str},
        )

        required = {
            "code",
            "effective_at",
            "short_type",
            "is_shortable",
            "short_restricted",
        }

        missing = required - set(
            frame.columns
        )

        if missing:
            raise ValueError(
                "shortability CSV missing "
                f"columns: {sorted(missing)}"
            )

        if "source" not in frame.columns:
            frame["source"] = (
                self.default_source
            )

        if "fetched_at" not in frame.columns:
            timestamp = pd.Timestamp(
                fetched_at
            )

            if timestamp.tzinfo is None:
                timestamp = (
                    timestamp.tz_localize(
                        "Asia/Tokyo"
                    )
                )

            frame["fetched_at"] = (
                timestamp
                .tz_convert("UTC")
                .isoformat()
            )

        if (
            "is_margin_lendable"
            not in frame.columns
        ):
            frame["is_margin_lendable"] = (
                pd.NA
            )

        if (
            "stock_loan_fee_annual"
            not in frame.columns
        ):
            frame[
                "stock_loan_fee_annual"
            ] = pd.NA

        return (
            normalize_shortability_observations(
                frame
            )
        )
