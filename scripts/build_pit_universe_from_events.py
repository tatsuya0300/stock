"""イベント履歴から point-in-time ユニバースCSVを構築する。

入力:
  - base CSV: code,name  (初期メンバー。effective期間なし)
  - events CSV: as_of,action(ADD|REMOVE),code,name

出力:
  - code,name,effective_from,effective_to

使い方:
  python scripts/build_pit_universe_from_events.py \
    --base data/topix500_base.csv \
    --events data/universe_events.csv \
    --out data/topix500.csv \
    --start 2022-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jp_signal.universe import normalize_code


def load_base(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].map(normalize_code)
    if "name" not in df.columns:
        df["name"] = ""
    return df[["code", "name"]].drop_duplicates("code")


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    need = {"as_of", "action", "code"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    df["code"] = df["code"].map(normalize_code)
    df["action"] = df["action"].astype(str).str.upper()
    df["as_of"] = pd.to_datetime(df["as_of"])
    if "name" not in df.columns:
        df["name"] = ""
    bad = set(df["action"].unique()) - {"ADD", "REMOVE"}
    if bad:
        raise ValueError(f"unknown actions: {sorted(bad)}")
    return df.sort_values(["as_of", "action", "code"]).reset_index(drop=True)


def build_intervals(
    base: pd.DataFrame,
    events: pd.DataFrame,
    *,
    start: str,
    end: str = "2099-12-31",
) -> pd.DataFrame:
    """各銘柄の在籍区間を作る。

    仮定:
    - start 時点のメンバー = base
    - events は start 以降の変更
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # code -> name
    names = dict(zip(base["code"], base["name"], strict=False))
    for _, e in events.iterrows():
        if e["name"]:
            names[str(e["code"])] = e["name"]

    active: set[str] = set(base["code"].tolist())
    # code -> current open start date
    open_from: dict[str, pd.Timestamp] = {c: start_ts for c in active}
    intervals: list[dict] = []

    for _, e in events.iterrows():
        d = e["as_of"]
        code = str(e["code"])
        act = e["action"]

        if d < start_ts:
            continue
        if d > end_ts:
            break

        if act == "ADD":
            if code not in active:
                active.add(code)
                open_from[code] = d
        elif act == "REMOVE":
            if code in active:
                fr = open_from.get(code, start_ts)
                to = d - pd.Timedelta(days=1)
                if fr <= to:
                    intervals.append(
                        {
                            "code": code,
                            "name": names.get(code, ""),
                            "effective_from": fr.strftime("%Y-%m-%d"),
                            "effective_to": to.strftime("%Y-%m-%d"),
                        }
                    )
                active.remove(code)

    # 残存メンバーを end まで
    for code in sorted(active):
        fr = open_from[code]
        intervals.append(
            {
                "code": code,
                "name": names.get(code, ""),
                "effective_from": fr.strftime("%Y-%m-%d"),
                "effective_to": end_ts.strftime("%Y-%m-%d"),
            }
        )

    out = pd.DataFrame(intervals)
    if out.empty:
        return out
    return out.sort_values(["code", "effective_from"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--out", default="data/topix500.csv")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2099-12-31")
    args = p.parse_args()

    base = load_base(args.base)
    events = load_events(args.events)
    univ = build_intervals(base, events, start=args.start, end=args.end)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    univ.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out} rows={len(univ)} unique_codes={univ['code'].nunique()}")


if __name__ == "__main__":
    main()
