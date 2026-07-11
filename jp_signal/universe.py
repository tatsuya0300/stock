"""ユニバース管理（FR-UNIV-01/02/03）。

TOPIX500構成銘柄をCSVで管理する。

対応形式:
- 静的形式:
    code,name
- point-in-time形式:
    code,name,effective_from,effective_to

effective_from / effective_to は両端を含む。
同一コードに複数の非重複期間を設定できる。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DEFAULT_EFFECTIVE_FROM = pd.Timestamp("1900-01-01")
_DEFAULT_EFFECTIVE_TO = pd.Timestamp("2099-12-31")


def normalize_code(code: str) -> str:
    """証券コードを正規化する。

    - 末尾の取引所サフィックスを除去する。
    - 数値コードは最低4桁にゼロ埋めする。
    """
    c = str(code).strip().upper()

    for suffix in (".T", ".TKO", ".N", ".L", ".O"):
        if c.endswith(suffix):
            c = c[: -len(suffix)]
            break

    try:
        n = int(c)
    except ValueError:
        return c

    return f"{n:04d}"


def _parse_effective_column(
    df: pd.DataFrame,
    column: str,
    default: pd.Timestamp,
) -> pd.Series:
    """有効期間列を正規化する。空欄はdefaultとする。"""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="datetime64[ns]")

    # NaN/空文字をdefault文字列で置き換えた上でパースする
    raw = df[column].fillna("").astype(str).str.strip()
    raw = raw.where(raw.ne(""), default.strftime("%Y-%m-%d"))

    parsed = pd.to_datetime(raw, errors="coerce").dt.normalize()

    invalid = parsed.isna()
    if invalid.any():
        rows = (df.index[invalid] + 2).tolist()
        raise ValueError(f"{column} に解釈不能な日付があります。CSV行: {rows}")

    return parsed


def _validate_effective_intervals(df: pd.DataFrame) -> None:
    """同一コードの有効期間が重複していないことを検証する。"""
    invalid_range = df["_effective_from"] > df["_effective_to"]
    if invalid_range.any():
        rows = (df.index[invalid_range] + 2).tolist()
        raise ValueError(f"effective_from が effective_to より後です。 CSV行: {rows}")

    overlaps: list[str] = []

    for code, group in df.groupby("code", sort=False):
        ordered = group.sort_values(
            ["_effective_from", "_effective_to"],
            kind="stable",
        )

        previous_to: pd.Timestamp | None = None

        for _, row in ordered.iterrows():
            current_from = pd.Timestamp(row["_effective_from"])
            current_to = pd.Timestamp(row["_effective_to"])

            # effective_toはinclusiveなので、同日開始は重複。
            if previous_to is not None and current_from <= previous_to:
                overlaps.append(str(code))
                break

            previous_to = current_to

    if overlaps:
        raise ValueError(f"有効期間が重複しています: {overlaps}")


def load_universe(
    path_or_cfg: str | dict,
    as_of: str | None = None,
) -> pd.DataFrame:
    """CSVからユニバースを読み込む。

    as_of指定時:
        指定日時点で有効なcode,nameを返す。

    as_of未指定時:
        全期間に登場するcodeの集合を返す。
        同一コードに複数期間がある場合、最新期間のnameを採用する。
    """
    if isinstance(path_or_cfg, dict):
        path = path_or_cfg.get("file")
        if not path:
            raise ValueError("universe.file が設定されていません")
    else:
        path = path_or_cfg

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ユニバースCSVが見つかりません: {path}. "
            "JPX公式データを基にcode,name列を用意してください。"
        )

    df = pd.read_csv(p, dtype={"code": str})

    if "code" not in df.columns:
        raise ValueError("ユニバースCSVに 'code' 列が必要です。")

    if "name" not in df.columns:
        df["name"] = ""

    if df["code"].isna().any():
        rows = (df.index[df["code"].isna()] + 2).tolist()
        raise ValueError(f"codeが空です。CSV行: {rows}")

    df["code"] = df["code"].map(normalize_code)
    df["name"] = df["name"].fillna("").astype(str).str.strip()

    has_pit_columns = "effective_from" in df.columns or "effective_to" in df.columns

    if not has_pit_columns:
        dupes = df[df.duplicated(subset="code", keep=False)]
        if not dupes.empty:
            raise ValueError(
                f"静的ユニバースに重複コードがあります: {sorted(dupes['code'].unique().tolist())}"
            )

        return df[["code", "name"]].sort_values("code").reset_index(drop=True)

    df["_effective_from"] = _parse_effective_column(
        df,
        "effective_from",
        _DEFAULT_EFFECTIVE_FROM,
    )
    df["_effective_to"] = _parse_effective_column(
        df,
        "effective_to",
        _DEFAULT_EFFECTIVE_TO,
    )

    _validate_effective_intervals(df)

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of).normalize()

        active = df[(df["_effective_from"] <= as_of_ts) & (df["_effective_to"] >= as_of_ts)].copy()

        # interval validator通過後なので通常は発生しないが、防御的に確認。
        dupes = active[active.duplicated(subset="code", keep=False)]
        if not dupes.empty:
            raise ValueError(
                f"as_of時点で複数の有効期間が存在します: {sorted(dupes['code'].unique().tolist())}"
            )

        return active[["code", "name"]].sort_values("code").reset_index(drop=True)

    # 全期間の取得対象コードを返す。
    # nameは最新期間の値を採用する。
    latest = (
        df.sort_values(
            ["code", "_effective_from", "_effective_to"],
            kind="stable",
        )
        .groupby("code", as_index=False)
        .tail(1)
    )

    return latest[["code", "name"]].sort_values("code").reset_index(drop=True)
