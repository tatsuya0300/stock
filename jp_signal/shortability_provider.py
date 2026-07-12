"""空売り可能データソース抽象化＋CSV実装。

PR-2: ShortabilityDataSource を導入し、shortability 情報の
取得元を抽象化する。CsvShortabilityDataSource が CSV ファイルから
読み込む最小実装を提供する。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import pandas as pd

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
