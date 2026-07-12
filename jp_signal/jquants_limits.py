"""J-Quantsのプラン別リクエスト間隔設定。

レート制限に関する知識をdatasource/config/testへ重複させず、
このモジュールに集約する。

注意:
    J-Quantsのレート制限は変更される可能性がある。
    マージ前および定期的に公式仕様を確認すること。

公式:
    https://jpx-jquants.com/en/spec/rate-limit
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

JQUANTS_PLAN_MIN_INTERVAL_SEC: Final[Mapping[str, float]] = MappingProxyType(
    {
        "free": 12.0,
        "light": 1.0,
        "standard": 0.5,
        "premium": 0.2,
    }
)

JQUANTS_MAX_INTERVAL_SEC: Final[float] = 120.0


def normalize_jquants_plan(plan: str) -> str:
    """J-Quantsプラン名を正規化・検証する。"""
    normalized = str(plan).strip().lower()

    if normalized not in JQUANTS_PLAN_MIN_INTERVAL_SEC:
        valid = sorted(JQUANTS_PLAN_MIN_INTERVAL_SEC)
        raise ValueError(
            "J-Quants plan must be one of "
            f"{valid}: {plan!r}"
        )

    return normalized


def resolve_jquants_sleep_sec(
    *,
    plan: str,
    sleep_sec: float | None,
) -> tuple[str, float]:
    """プランとリクエスト間隔を正規化する。

    Args:
        plan:
            J-Quantsの契約プラン名。

        sleep_sec:
            リクエスト間隔。Noneの場合は、プラン別の
            最小安全間隔を自動採用する。

    Returns:
        (正規化済みplan, 確定したsleep_sec)

    Raises:
        ValueError:
            planが不正、sleep_secがプラン下限未満、
            またはシステム上限を超える場合。
    """
    normalized_plan = normalize_jquants_plan(plan)

    minimum = float(JQUANTS_PLAN_MIN_INTERVAL_SEC[normalized_plan])

    resolved = minimum if sleep_sec is None else float(sleep_sec)

    if resolved < minimum:
        raise ValueError(
            f"sleep_sec ({resolved}) must be >= "
            f"{normalized_plan} plan minimum "
            f"({minimum})"
        )

    if resolved > JQUANTS_MAX_INTERVAL_SEC:
        raise ValueError(
            f"sleep_sec ({resolved}) must be <= "
            f"{JQUANTS_MAX_INTERVAL_SEC}"
        )

    return normalized_plan, resolved
