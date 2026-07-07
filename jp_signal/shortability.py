"""売り可否（日証金スナップショット）管理（FR-DATA-04）。

日証金は「最新スナップショット」のみ公開のため、日次で取得して蓄積する。
過去分は不明扱い（is_margin_lendable=NULL）。FR-BT-05 に従い売り戦略BTから除外する。
一次情報: https://www.jsf.co.jp/

注記（忖度なし）:
  日証金の公開データはCSV/HTMLのフォーマットが変わり得るため、
  snapshot_today() の本実装前に必ず一次情報でフォーマットを確認すること。
  MVPでは "TOPIX500は概ね貸借銘柄" という前提で暫定的に埋め、
  short_restricted は未取得時に保守側（=売り不可扱い）に倒すのが安全。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

SHORT_COLS = ["code", "date", "is_margin_lendable", "short_restricted"]


class ShortabilityProvider:
    """日証金スナップショットの取得・整形を担う。"""

    def snapshot_today(self, today: date) -> pd.DataFrame:
        """当日の売り可否スナップショットを返す（本実装は一次情報確認後）。

        returns columns: code, date, is_margin_lendable(0/1), short_restricted(0/1)
        """
        raise NotImplementedError(
            "日証金の公開データ仕様に合わせて実装する（https://www.jsf.co.jp/）"
        )

    def provisional_snapshot(
        self, codes: list[str], today: date, assume_shortable: bool = False
    ) -> pd.DataFrame:
        """MVP向けの暫定スナップショット。

        TOPIX500は概ね貸借銘柄との前提で is_margin_lendable=1 とする。

        assume_shortable=False（デフォルト）: short_restricted=1（売り不可）に倒し、
            売り戦略PnLの保守的下限を推定する。実データが無い間はこちらが安全側。
        assume_shortable=True: 開発時の疎通確認用に楽観的に売り可能（short_restricted=0）
            とする。実運用・実データ検証では使用しないこと。

        本メソッドはあくまで開発・検証用であり、実際の売り可否を保証しない。
        実運用では snapshot_today() で置き換えること。
        """
        d = today.strftime("%Y-%m-%d")
        restricted = 0 if assume_shortable else 1
        return pd.DataFrame(
            {
                "code": codes,
                "date": [d] * len(codes),
                "is_margin_lendable": [1] * len(codes),
                "short_restricted": [restricted] * len(codes),
            }
        )[SHORT_COLS]
