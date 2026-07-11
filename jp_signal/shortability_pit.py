"""Point-in-time shortability observations.

このモジュールは既存のshortability.pyと並行して使用する。

設計原則:
- effective_at: 売り可否情報が効力を持つ時刻
- fetched_at: システムが実際に取得した時刻
- available_at: max(effective_at, fetched_at)
- decision時点でavailable_at <= as_ofの観測のみ使用
- future dataを参照しない
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

ShortType = Literal["system", "general", "unknown"]

VALID_SHORT_TYPES: set[str] = {"system", "general", "unknown"}

REQUIRED_COLUMNS = {
    "code",
    "effective_at",
    "fetched_at",
    "source",
    "short_type",
    "is_shortable",
    "short_restricted",
}

OUTPUT_COLUMNS = [
    "code",
    "effective_at",
    "fetched_at",
    "available_at",
    "source",
    "short_type",
    "is_shortable",
    "is_margin_lendable",
    "short_restricted",
    "stock_loan_fee_annual",
    "payload_hash",
]


@dataclass(frozen=True)
class ShortabilityDecision:
    """売り可否判定の結果。"""

    code: str
    as_of: pd.Timestamp
    is_shortable: bool
    short_type: str
    reason: str
    source: str | None = None
    effective_at: pd.Timestamp | None = None
    fetched_at: pd.Timestamp | None = None
    available_at: pd.Timestamp | None = None
    stock_loan_fee_annual: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "as_of": self.as_of.isoformat(),
            "is_shortable": self.is_shortable,
            "short_type": self.short_type,
            "reason": self.reason,
            "source": self.source,
            "effective_at": self.effective_at.isoformat() if self.effective_at else None,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "available_at": self.available_at.isoformat() if self.available_at else None,
            "stock_loan_fee_annual": self.stock_loan_fee_annual,
        }


def _parse_binary_value(
    value: Any,
    *,
    column: str,
    allow_null: bool = False,
) -> int | None:
    """0/1または一般的なbool表現を厳密に変換する。"""
    if value is None or pd.isna(value):
        if allow_null:
            return None
        raise ValueError(f"{column} must not be null")

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int | float):
        if float(value) == 0.0:
            return 0
        if float(value) == 1.0:
            return 1
        raise ValueError(f"{column} must be 0 or 1: {value!r}")

    normalized = str(value).strip().lower()

    true_values = {
        "1",
        "true",
        "yes",
        "y",
        "可",
        "可能",
    }
    false_values = {
        "0",
        "false",
        "no",
        "n",
        "不可",
        "禁止",
    }

    if normalized in true_values:
        return 1

    if normalized in false_values:
        return 0

    raise ValueError(f"{column} has invalid binary value: {value!r}")


def _normalize_timestamp(
    values: pd.Series,
    *,
    column: str,
) -> pd.Series:
    """timestampをUTCへ統一する。"""
    try:
        result = pd.to_datetime(
            values,
            errors="raise",
            utc=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} contains invalid timestamp") from exc

    if result.isna().any():
        raise ValueError(f"{column} must not contain null")

    return result


def _payload_hash(row: dict[str, Any]) -> str:
    """観測内容の決定論的hashを作る。"""
    payload = {
        key: (
            value.isoformat()
            if isinstance(value, pd.Timestamp)
            else value
        )
        for key, value in row.items()
        if key != "payload_hash"
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def normalize_shortability_observations(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """shortability観測を正規化・検証する。"""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    missing = REQUIRED_COLUMNS - set(frame.columns)

    if missing:
        raise ValueError(
            "shortability observations missing columns: "
            f"{sorted(missing)}"
        )

    x = frame.copy()

    x["code"] = (
        x["code"]
        .astype("string")
        .str.strip()
    )

    if x["code"].isna().any() or x["code"].eq("").any():
        raise ValueError("code must not be empty")

    x["effective_at"] = _normalize_timestamp(
        x["effective_at"],
        column="effective_at",
    )
    x["fetched_at"] = _normalize_timestamp(
        x["fetched_at"],
        column="fetched_at",
    )

    # 事前公表され、将来発効する情報もあり得る。
    # したがってfetched_at >= effective_atは要求しない。
    x["available_at"] = pd.concat(
        [
            x["effective_at"],
            x["fetched_at"],
        ],
        axis=1,
    ).max(axis=1)

    x["source"] = (
        x["source"]
        .astype("string")
        .str.strip()
    )

    if x["source"].isna().any() or x["source"].eq("").any():
        raise ValueError("source must not be empty")

    x["short_type"] = (
        x["short_type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    invalid_short_types = (
        set(x["short_type"].dropna())
        - VALID_SHORT_TYPES
    )

    if invalid_short_types:
        raise ValueError(
            "invalid short_type: "
            f"{sorted(invalid_short_types)}"
        )

    x["is_shortable"] = [
        _parse_binary_value(
            value,
            column="is_shortable",
        )
        for value in x["is_shortable"]
    ]

    x["short_restricted"] = [
        _parse_binary_value(
            value,
            column="short_restricted",
        )
        for value in x["short_restricted"]
    ]

    if "is_margin_lendable" not in x.columns:
        x["is_margin_lendable"] = pd.NA

    x["is_margin_lendable"] = [
        _parse_binary_value(
            value,
            column="is_margin_lendable",
            allow_null=True,
        )
        for value in x["is_margin_lendable"]
    ]

    x["is_shortable"] = pd.Series(
        x["is_shortable"],
        index=x.index,
        dtype="Int64",
    )
    x["short_restricted"] = pd.Series(
        x["short_restricted"],
        index=x.index,
        dtype="Int64",
    )
    x["is_margin_lendable"] = pd.Series(
        x["is_margin_lendable"],
        index=x.index,
        dtype="Int64",
    )

    if "stock_loan_fee_annual" not in x.columns:
        x["stock_loan_fee_annual"] = pd.NA

    x["stock_loan_fee_annual"] = pd.to_numeric(
        x["stock_loan_fee_annual"],
        errors="coerce",
    )

    negative_fee = (
        x["stock_loan_fee_annual"].notna()
        & (x["stock_loan_fee_annual"] < 0)
    )

    if negative_fee.any():
        raise ValueError(
            "stock_loan_fee_annual must be >= 0"
        )

    # 制度信用では貸借対象情報を必須とする。
    system_rows = x["short_type"] == "system"

    if x.loc[
        system_rows,
        "is_margin_lendable",
    ].isna().any():
        raise ValueError(
            "system shortability requires "
            "is_margin_lendable"
        )

    hashes: list[str] = []

    for _, row in x.iterrows():
        hashes.append(
            _payload_hash(
                {
                    "code": str(row["code"]),
                    "effective_at": row["effective_at"],
                    "fetched_at": row["fetched_at"],
                    "available_at": row["available_at"],
                    "source": str(row["source"]),
                    "short_type": str(row["short_type"]),
                    "is_shortable": int(
                        row["is_shortable"]
                    ),
                    "is_margin_lendable": (
                        None
                        if pd.isna(
                            row["is_margin_lendable"]
                        )
                        else int(
                            row[
                                "is_margin_lendable"
                            ]
                        )
                    ),
                    "short_restricted": int(
                        row["short_restricted"]
                    ),
                    "stock_loan_fee_annual": (
                        None
                        if pd.isna(
                            row[
                                "stock_loan_fee_annual"
                            ]
                        )
                        else float(
                            row[
                                "stock_loan_fee_annual"
                            ]
                        )
                    ),
                }
            )
        )

    x["payload_hash"] = hashes

    return (
        x[OUTPUT_COLUMNS]
        .sort_values(
            [
                "code",
                "available_at",
                "effective_at",
                "fetched_at",
                "source",
                "short_type",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def load_shortability_observations_csv(
    path: str | Path,
) -> pd.DataFrame:
    """正規化済みCSVを読み込む。"""
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"shortability CSV not found: {path}"
        )

    frame = pd.read_csv(
        csv_path,
        dtype={"code": str},
    )

    return normalize_shortability_observations(
        frame
    )


def _normalize_as_of(
    as_of: str | pd.Timestamp,
    *,
    assume_timezone: str = "Asia/Tokyo",
) -> pd.Timestamp:
    """decision timestampをUTCへ変換する。

    timezoneなしの場合はassume_timezoneとして解釈する。
    """
    timestamp = pd.Timestamp(as_of)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            assume_timezone
        )

    return timestamp.tz_convert("UTC")


def decide_shortability(
    observations: pd.DataFrame | None,
    *,
    code: str,
    as_of: str | pd.Timestamp,
    requested_short_type: ShortType = "system",
    max_age: pd.Timedelta | None = None,
    assume_timezone: str = "Asia/Tokyo",
) -> ShortabilityDecision:
    """指定時点で利用可能だった直近観測から売り可否を決定する。"""
    if max_age is None:
        max_age = pd.Timedelta(days=4)

    if max_age < pd.Timedelta(days=0):
        raise ValueError(
            f"max_age must be >= 0: {max_age}"
        )

    if requested_short_type not in VALID_SHORT_TYPES:
        raise ValueError(
            "invalid requested_short_type: "
            f"{requested_short_type}"
        )

    as_of_utc = _normalize_as_of(
        as_of,
        assume_timezone=assume_timezone,
    )

    normalized_code = str(code).strip()

    def rejected(
        reason: str,
        *,
        row: pd.Series | None = None,
    ) -> ShortabilityDecision:
        return ShortabilityDecision(
            code=normalized_code,
            as_of=as_of_utc,
            is_shortable=False,
            short_type=(
                requested_short_type
                if row is None
                else str(row["short_type"])
            ),
            reason=reason,
            source=(
                None
                if row is None
                else str(row["source"])
            ),
            effective_at=(
                None
                if row is None
                else pd.Timestamp(
                    row["effective_at"]
                )
            ),
            fetched_at=(
                None
                if row is None
                else pd.Timestamp(
                    row["fetched_at"]
                )
            ),
            available_at=(
                None
                if row is None
                else pd.Timestamp(
                    row["available_at"]
                )
            ),
            stock_loan_fee_annual=(
                None
                if row is None
                or pd.isna(
                    row["stock_loan_fee_annual"]
                )
                else float(
                    row["stock_loan_fee_annual"]
                )
            ),
        )

    if observations is None or observations.empty:
        return rejected("NO_OBSERVATION")

    x = normalize_shortability_observations(
        observations
    )

    eligible = x[
        (x["code"] == normalized_code)
        & (x["short_type"] == requested_short_type)
        & (x["available_at"] <= as_of_utc)
        ]

    if eligible.empty:
        return rejected("NO_AVAILABLE_OBSERVATION")

    latest = eligible.sort_values(
        ["available_at", "effective_at", "fetched_at"],
    ).iloc[-1]

    age = as_of_utc - latest["available_at"]

    if age > max_age:
        return rejected(
            "STALE_OBSERVATION",
            row=latest,
        )

    if int(latest["short_restricted"]) != 0:
        return rejected(
            "SHORT_RESTRICTED",
            row=latest,
        )

    if int(latest["is_shortable"]) != 1:
        return rejected(
            "NOT_SHORTABLE",
            row=latest,
        )

    if requested_short_type == "system":
        if pd.isna(latest["is_margin_lendable"]):
            return rejected(
                "MARGIN_LENDABILITY_UNKNOWN",
                row=latest,
            )

        if int(latest["is_margin_lendable"]) != 1:
            return rejected(
                "NOT_MARGIN_LENDABLE",
                row=latest,
            )

    if requested_short_type == "unknown":
        return rejected(
            "UNKNOWN_SHORT_TYPE",
            row=latest,
        )

    return ShortabilityDecision(
        code=normalized_code,
        as_of=as_of_utc,
        is_shortable=True,
        short_type=str(latest["short_type"]),
        reason="SHORTABLE",
        source=str(latest["source"]),
        effective_at=pd.Timestamp(
            latest["effective_at"]
        ),
        fetched_at=pd.Timestamp(
            latest["fetched_at"]
        ),
        available_at=pd.Timestamp(
            latest["available_at"]
        ),
        stock_loan_fee_annual=(
            None
            if pd.isna(
                latest["stock_loan_fee_annual"]
            )
            else float(
                latest["stock_loan_fee_annual"]
            )
        ),
    )
