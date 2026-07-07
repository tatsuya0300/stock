"""ユニバース管理（FR-UNIV-01/02/03）。

TOPIX500 構成銘柄を CSV (code,name) で管理する。
一次情報: https://www.jpx.co.jp/markets/indices/topix/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_universe(path: str) -> pd.DataFrame:
    """CSV: code,name。code は文字列（先頭ゼロ保持）で読み込む。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ユニバースCSVが見つかりません: {path}. "
            "JPX公式のTOPIX500構成銘柄を code,name の2列で用意してください。"
        )
    df = pd.read_csv(p, dtype={"code": str})
    if "code" not in df.columns:
        raise ValueError("ユニバースCSVに 'code' 列が必要です。")
    if "name" not in df.columns:
        df["name"] = ""
    df["code"] = df["code"].str.strip()
    return df[["code", "name"]]
