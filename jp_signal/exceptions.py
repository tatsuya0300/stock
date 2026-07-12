"""カスタム例外クラス。"""

from __future__ import annotations


class JQuantsError(Exception):
    """J-Quants API関連のエラーの基底クラス。"""


class AuthenticationError(JQuantsError):
    """認証エラー（401/403）。"""


class RateLimitError(JQuantsError):
    """レート制限エラー（429）。"""


class RequestError(JQuantsError):
    """その他のHTTPリクエストエラー。"""


class ResponseSchemaError(JQuantsError):
    """レスポンスのスキーマが期待と異なる場合。"""
