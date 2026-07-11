"""バックテスト入力データのスナップショット。

バックテストに使った入力をCSV＋SHA-256で保存し、再現性を担保する。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256_file(path: Path) -> str:
    """ファイルのSHA-256を計算する。"""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """決定論的な順序へ正規化する。"""
    if frame is None or frame.empty:
        return pd.DataFrame()

    output = frame.copy()
    columns = sorted(str(column) for column in output.columns)
    output = output[columns]

    preferred_sort_columns = [
        column
        for column in [
            "code",
            "date",
            "available_at",
            "effective_at",
            "fetched_at",
            "short_type",
            "source",
        ]
        if column in output.columns
    ]

    if preferred_sort_columns:
        output = output.sort_values(preferred_sort_columns, kind="stable")

    return output.reset_index(drop=True)


def _write_frame(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    canonical = _canonical_frame(frame)

    canonical.to_csv(
        path,
        index=False,
        lineterminator="\n",
        date_format="%Y-%m-%dT%H:%M:%S%z",
        float_format="%.17g",
    )

    return {
        "file": path.name,
        "rows": len(canonical),
        "columns": list(canonical.columns),
        "sha256": _sha256_file(path),
    }


def write_backtest_input_snapshot(
    *,
    output_dir: str | Path,
    prices: pd.DataFrame,
    shortability: pd.DataFrame | None,
    universe_file: str | Path,
) -> dict[str, Any]:
    """BT入力をCSVとSHA-256で保存する。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "format_version": 1,
    }

    metadata["prices"] = _write_frame(prices, output / "prices.csv")

    metadata["shortability"] = _write_frame(
        (shortability if shortability is not None else pd.DataFrame()),
        output / "shortability.csv",
    )

    source_universe = Path(universe_file)
    if not source_universe.exists():
        raise FileNotFoundError(f"universe file not found: {source_universe}")

    snapshot_universe = output / "universe.csv"
    shutil.copyfile(source_universe, snapshot_universe)

    metadata["universe"] = {
        "file": snapshot_universe.name,
        "sha256": _sha256_file(snapshot_universe),
    }

    metadata_path = output / "snapshot.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata["metadata_sha256"] = _sha256_file(metadata_path)

    return metadata
