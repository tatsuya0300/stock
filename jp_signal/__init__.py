"""jp_signal: 日本株シグナル生成 + バックテスト MVP パッケージ.

FR-DATA（データ取得基盤）と FR-BT（バックテスト）の最小実装を提供する。
FR-NOTIFY はアダプタパターンで後段接続可能な設計。
"""

import pandas as pd

__version__ = "0.1.0"


def normalize_code(code: str) -> str:
    """日本株コードを正規化する（4桁統一）。

    - 数字4桁未満は先頭ゼロ埋めしない
    - 数字5桁以上は4桁に切り詰めない（J-Quants 5桁コードなどは維持）
    - 空白除去、文字列化を行う

    Args:
        code: 入力コード（int, float, str 可）

    Returns:
        正規化されたコード文字列
    """
    s = str(code).strip()
    # J-Quants の 5桁コード（例: "86970"）はそのまま維持
    # 通常の4桁コードはそのまま
    return s


def normalize_date(dt: str | pd.Timestamp) -> str:
    """日付を YYYY-MM-DD 形式に正規化する。

    Args:
        dt: 入力日付（文字列、pd.Timestamp）

    Returns:
        YYYY-MM-DD 形式の日付文字列
    """
    return pd.Timestamp(dt).strftime("%Y-%m-%d")
