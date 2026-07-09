"""JPX 公開ウェイトファイルからユニバースCSVを生成する。

使い方:
  python scripts/build_universe_from_jpx_weights.py --top 500 --out data/topix500.csv

注意:
  - 最新構成ベースのため生存者バイアスが残る。
  - historical point-in-time ではない。
  - 厳密な PIT が必要なら定期選定の ADD/REMOVE を events CSV で管理し、
    scripts/build_pit_universe_from_events.py を使うこと。
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from jp_signal.universe import normalize_code

# JPX ウェイトファイル（TOPIX500）
DEFAULT_URL = "https://www.jpx.co.jp/markets/indices/topix/weight/nlsgeu000006pnsu-data.csv"


def fetch_jpx_weights(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    # JPX CSV は Shift_JIS のことが多い
    text = r.content.decode("cp932", errors="replace")
    df = pd.read_csv(StringIO(text))
    return df


def normalize_weight_frame(df: pd.DataFrame) -> pd.DataFrame:
    """列名ゆれを吸収して code,name,weight に正規化。"""
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)

    code_candidates = ["コード", "銘柄コード", "code", "Code"]
    name_candidates = ["銘柄名", "name", "Name"]
    weight_candidates = ["ウエイト", "ウェイト", "weight", "Weight", "構成比率"]

    def pick(cands: list[str]) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        # 部分一致
        for col in df.columns:
            for c in cands:
                if c.lower() in str(col).lower():
                    return col
        return None

    code_col = pick(code_candidates)
    name_col = pick(name_candidates)
    weight_col = pick(weight_candidates)

    if code_col is None:
        raise ValueError(f"code 列を検出できない。columns={list(df.columns)}")

    out = pd.DataFrame()
    out["code"] = df[code_col].astype(str).map(normalize_code)
    out["name"] = df[name_col].astype(str) if name_col else ""
    if weight_col:
        out["weight"] = pd.to_numeric(
            df[weight_col].astype(str).str.replace("%", ""),
            errors="coerce",
        )
    else:
        out["weight"] = pd.NA

    out = out.dropna(subset=["code"])
    out = out[out["code"].str.len() > 0]
    out = out.drop_duplicates(subset=["code"], keep="first")
    return out


def to_universe(
    df: pd.DataFrame,
    *,
    top: int | None,
    effective_from: str,
    effective_to: str,
) -> pd.DataFrame:
    x = df.copy()
    if top is not None:
        if x["weight"].notna().any():
            x = x.sort_values("weight", ascending=False).head(int(top))
        else:
            x = x.head(int(top))

    out = pd.DataFrame(
        {
            "code": x["code"].tolist(),
            "name": x["name"].tolist(),
            "effective_from": effective_from,
            "effective_to": effective_to,
        }
    )
    return out.sort_values("code").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out", default="data/topix500.csv")
    p.add_argument("--top", type=int, default=500, help="上位N件。0で全件")
    p.add_argument("--effective-from", default="1900-01-01")
    p.add_argument("--effective-to", default="2099-12-31")
    args = p.parse_args()

    raw = fetch_jpx_weights(args.url)
    norm = normalize_weight_frame(raw)
    top = None if args.top == 0 else args.top
    univ = to_universe(
        norm,
        top=top,
        effective_from=args.effective_from,
        effective_to=args.effective_to,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    univ.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out} rows={len(univ)}")
    print(
        "WARNING: これは最新構成ベースです。"
        "historical point-in-time ではありません（生存者バイアス残存）。"
    )


if __name__ == "__main__":
    main()
