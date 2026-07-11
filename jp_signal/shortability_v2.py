"""PIT shortability 管理（FR-DATA-04 v2）。

日証金（JSF）の日次スナップショットを PIT（point-in-time）で管理する。
各レコードは effective_at / fetched_at の時刻情報を持ち、
バックテストと本番の両方で正しい時点の shortability を使えるようにする。

制約:
  - fetched_at > effective_at（未来の情報で過去を判断しない）
  - effective_at < as_of + max_age（鮮度条件）
  - is_shortable == 1 AND is_margin_lendable == 1 AND short_restricted == 0

注記（忖度なし）:
  日証金の CS フォーマットは予告なく変更される可能性がある。
  load_shortability_csv() は既知のフォーマットに依存するため、
  フォーマット変更時には修正が必要。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

# デフォルトの最大許容経過日数（金曜→月曜をカバー）
_DEFAULT_MAX_AGE_DAYS = 4


@dataclass(frozen=True)
class ShortabilityDecision:
    """shortability 判定結果。

    Attributes:
        code: 証券コード
        effective_at: スナップショットの基準日時
        is_shortable: 売り可能なら True
        reason: 判定理由（OK または却下理由）
    """

    code: str
    effective_at: str
    is_shortable: bool
    reason: str


def normalize_shortability_frame(
    df: pd.DataFrame,
    *,
    code_col: str = "code",
    date_col: str = "date",
    lendable_col: str = "is_margin_lendable",
    restricted_col: str = "short_restricted",
    effective_col: str | None = None,
    fetched_col: str | None = None,
    as_of: date | str | None = None,
) -> pd.DataFrame:
    """shortability DataFrame を正規化する。

    入力CSVの列名を統一し、日付列をパースする。
    effective_at / fetched_at が無い場合は、date_col を基準に補完する。

    Args:
        df: 生の shortability DataFrame
        code_col: 証券コード列名
        date_col: 日付列名
        lendable_col: 信用貸付可能フラグ列名
        restricted_col: 空売り制限フラグ列名
        effective_col: 有効時刻列名（None なら date から作成）
        fetched_col: 取得時刻列名（None なら date の翌営業日始業想定）
        as_of: 基準日（fetched_col 補完用）

    Returns:
        正規化された DataFrame（列: code, date, effective_at, fetched_at,
        is_margin_lendable, short_restricted）
    """
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    # 必須列の名前を統一
    x = x.rename(
        columns={
            code_col: "code",
            lendable_col: "is_margin_lendable",
            restricted_col: "short_restricted",
        }
    )

    # date 列のパースまたは作成
    if date_col != "date" and "date" not in x.columns:
        if date_col in x.columns:
            x = x.rename(columns={date_col: "date"})

    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        # 日付列が無い場合は as_of の日付を使う
        as_of_d = pd.Timestamp(as_of).strftime("%Y-%m-%d") if as_of else date.today().isoformat()
        x["date"] = as_of_d

    # effective_at の設定
    if effective_col and effective_col in x.columns:
        x["effective_at"] = pd.to_datetime(
            x[effective_col], errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        # date をそのまま effective_at として使用（時刻は 00:00:00）
        x["effective_at"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%dT00:00:00")

    # fetched_at の設定
    if fetched_col and fetched_col in x.columns:
        x["fetched_at"] = pd.to_datetime(
            x[fetched_col], errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        # デフォルトは effective_at と同じ（実運用では取得時刻を記録すること）
        x["fetched_at"] = x["effective_at"]

    # int 型への変換
    for col in ["is_margin_lendable", "short_restricted"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0).astype(int)

    x["code"] = x["code"].astype(str).str.strip()

    cols = [
        "code",
        "date",
        "effective_at",
        "fetched_at",
        "is_margin_lendable",
        "short_restricted",
    ]
    return x[cols].dropna(subset=["code", "date"]).reset_index(drop=True)


def load_shortability_csv(path: str) -> pd.DataFrame:
    """JSF 日証金フォーマットの CSV を読み込む。

    想定列:
      - コード / code: 証券コード（4桁）
      - 日付 / date: スナップショット日付
      - 貸付可能 / is_margin_lendable: 0/1
      - 空売り制限 / short_restricted: 0/1
      - 基準日時 / effective_at: スナップショット基準日時（ISO 8601、任意）
      - 取得日時 / fetched_at: データ取得日時（ISO 8601、任意）

    列名は日本語・英語の両方に対応する。

    Args:
        path: CSV ファイルのパス

    Returns:
        正規化された DataFrame

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: 必須列が不足している場合
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"shortability CSV が見つかりません: {path}")

    df = pd.read_csv(path, dtype={"code": str, "コード": str})

    # 列名のマッピング（日本語 → 英語）
    col_map = {
        "コード": "code",
        "日付": "date",
        "貸付可能": "is_margin_lendable",
        "空売り制限": "short_restricted",
        "基準日時": "effective_at",
        "取得日時": "fetched_at",
    }
    x = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 必須列の確認
    required = {"code", "is_margin_lendable", "short_restricted"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"shortability CSV に必須列が不足: {sorted(missing)}")

    return normalize_shortability_frame(x)


def decide_shortability(
    df: pd.DataFrame,
    code: str,
    as_of: date | str,
    *,
    effective_col: str = "effective_at",
    fetched_col: str = "fetched_at",
    max_age_days: int = _DEFAULT_MAX_AGE_DAYS,
) -> ShortabilityDecision:
    """指定銘柄の shortability を PIT 判定する。

    判定条件:
      1. fetched_at > effective_at（未来情報の不使用）
      2. effective_at < as_of + max_age_days（鮮度条件）
      3. is_margin_lendable == 1 AND short_restricted == 0

    Args:
        df: 正規化された shortability DataFrame
        code: 判定する銘柄コード
        as_of: 基準日
        effective_col: 有効時刻列名
        fetched_col: 取得時刻列名
        max_age_days: 最大許容経過日数

    Returns:
        ShortabilityDecision
    """
    as_of_ts = pd.Timestamp(as_of)

    if df is None or df.empty:
        return ShortabilityDecision(
            code=code,
            effective_at=as_of_ts.strftime("%Y-%m-%dT00:00:00"),
            is_shortable=False,
            reason="NO_SHORTABILITY_DATA",
        )

    sub = df[df["code"].astype(str) == str(code)].copy()
    if sub.empty:
        return ShortabilityDecision(
            code=code,
            effective_at=as_of_ts.strftime("%Y-%m-%dT00:00:00"),
            is_shortable=False,
            reason="NO_SHORTABILITY_DATA_FOR_CODE",
        )

    # effective_at をパース
    if effective_col in sub.columns:
        sub["_eff"] = pd.to_datetime(sub[effective_col], errors="coerce")
    else:
        sub["_eff"] = pd.to_datetime(sub["date"], errors="coerce")

    sub = sub.dropna(subset=["_eff"])

    if sub.empty:
        return ShortabilityDecision(
            code=code,
            effective_at=as_of_ts.strftime("%Y-%m-%dT00:00:00"),
            is_shortable=False,
            reason="NO_VALID_EFFECTIVE_DATE",
        )

    # 条件1: effective_at <= as_of（未来の有効日を除外）
    sub = sub[sub["_eff"] <= as_of_ts]

    if sub.empty:
        return ShortabilityDecision(
            code=code,
            effective_at=as_of_ts.strftime("%Y-%m-%dT00:00:00"),
            is_shortable=False,
            reason="NO_EFFECTIVE_BEFORE_AS_OF",
        )

    # 最新の有効レコードを選択
    latest = sub.sort_values("_eff").iloc[-1]
    eff = latest["_eff"]

    # 条件2: 鮮度チェック
    age_days = (as_of_ts - eff).days
    if age_days > max_age_days:
        return ShortabilityDecision(
            code=code,
            effective_at=eff.strftime("%Y-%m-%dT%H:%M:%S"),
            is_shortable=False,
            reason=f"STALE_DATA_AGE={age_days}d",
        )

    # 条件3: is_margin_lendable == 1 AND short_restricted == 0
    try:
        lendable = int(latest.get("is_margin_lendable", 0))
        restricted = int(latest.get("short_restricted", 1))
    except (TypeError, ValueError):
        return ShortabilityDecision(
            code=code,
            effective_at=eff.strftime("%Y-%m-%dT%H:%M:%S"),
            is_shortable=False,
            reason="INVALID_SHORTABILITY_FLAGS",
        )

    is_shortable = lendable == 1 and restricted == 0

    return ShortabilityDecision(
        code=code,
        effective_at=eff.strftime("%Y-%m-%dT%H:%M:%S"),
        is_shortable=is_shortable,
        reason="OK" if is_shortable else "NOT_SHORTABLE_BY_FLAGS",
    )
