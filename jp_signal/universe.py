"""ユニバース管理（FR-UNIV-01/02/03）。

TOPIX500 構成銘柄を CSV (code,name) で管理する。
一次情報: https://www.jpx.co.jp/markets/indices/topix/

改訂点（FR-UNIV-02/03）:
  - normalize_code(): 証券コードの正規化（".T" 除去、4桁ゼロ埋め）。
  - load_universe(): point-in-time フィルタリング（effective_from/effective_to）。
  - 重複コードを拒否してデータ品質を向上。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_code(code: str) -> str:
    """証券コードを正規化する。

    - 末尾の .T 等の証券取引所サフィックスを除去。
    - 3桁コードは先頭ゼロ埋めで4桁に（例: "123" → "0123"）。
    """
    c = code.strip().upper()
    # 既知のサフィックス除去
    for suffix in (".T", ".TKO", ".N", ".L", ".O"):
        if c.endswith(suffix):
            c = c[: -len(suffix)]
            break
    # 整数化して4桁ゼロ埋め
    try:
        n = int(c)
        return f"{n:04d}"
    except ValueError:
        return c


def load_universe(
    path: str,
    as_of: str | None = None,
) -> pd.DataFrame:
    """CSV からユニバースを読み込む。

    Args:
        path: CSV ファイル (code, name[, effective_from, effective_to])。
        as_of: 指定日時点の有効な銘柄のみにフィルタ。
               未指定なら全件（effective_from/effective_to が無い古いCSVも読める）。

    Returns:
        code, name の DataFrame。code は正規化・ソート済み。
    """
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

    df["code"] = df["code"].str.strip().apply(normalize_code)

    # 重複コードチェック
    dupes = df[df.duplicated(subset="code", keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"重複コード検出: {dupes['code'].unique().tolist()}"
        )

    # point-in-time フィルタリング（日付比較を厳密化）
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        if "effective_from" in df.columns:
            ef = pd.to_datetime(df["effective_from"].fillna("1900-01-01"))
            df = df[ef <= as_of_ts]
        if "effective_to" in df.columns:
            et = pd.to_datetime(df["effective_to"].fillna("2099-12-31"))
            df = df[et >= as_of_ts]

    df = df.sort_values("code").reset_index(drop=True)
    return df[["code", "name"]]
