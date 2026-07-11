"""ポートフォリオ単位バックテスター。

管理対象:
- cash
- open positions
- realized PnL
- carry cost
- daily NAV
- long/short/gross/net exposure
- rejected orders

執行モデル:
- MKT_OPENでエントリー
- holding_days営業日後のMKT_CLOSEで決済
- prior ADVを使ったsquare-root impact
- 金利・貸株料は暦日計上

未実装:
- 証券会社別信用保証金
- 追証・強制決済
- 配当・corporate action
- 部分約定
- 日中価格経路
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .adv import stock_adv_before
from .risk import RiskConfig

Side = Literal["BUY", "SELL"]


@dataclass
class Position:
    position_id: str
    code: str
    name: str
    side: Side
    qty: int
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp
    entry_price: float
    entry_adv: float
    entry_slippage_bp: float
    score: float
    last_carry_date: pd.Timestamp
    accrued_carry: float = 0.0
    accrued_carry_days: int = 0


@dataclass
class PortfolioResult:
    trades: pd.DataFrame
    rejected_orders: pd.DataFrame
    daily_ledger: pd.DataFrame
    open_positions: pd.DataFrame


class PortfolioBacktester:
    """ポートフォリオ状態を持つ日次バックテスター。"""

    def __init__(
        self,
        *,
        initial_capital: float,
        risk: RiskConfig,
        impact_k_bp: float = 30.0,
        annual_interest_rate: float = 0.02,
        annual_lending_rate: float = 0.02,
        commission_bp: float = 0.0,
        half_spread_bp: float = 0.0,
        adv_window: int = 20,
        min_adv_periods: int = 20,
        require_liquidity_data: bool = True,
    ):
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0: {initial_capital}")
        if adv_window < 1:
            raise ValueError(f"adv_window must be >= 1: {adv_window}")
        if min_adv_periods < 1:
            raise ValueError(f"min_adv_periods must be >= 1: {min_adv_periods}")
        if min_adv_periods > adv_window:
            raise ValueError(
                f"min_adv_periods ({min_adv_periods}) must be <= adv_window ({adv_window})"
            )

        self.initial_capital = float(initial_capital)
        self.risk = risk
        self.impact_k_bp = float(impact_k_bp)
        self.annual_interest_rate = float(annual_interest_rate)
        self.annual_lending_rate = float(annual_lending_rate)
        self.daily_interest = self.annual_interest_rate / 365.0
        self.daily_lending = self.annual_lending_rate / 365.0
        self.commission_bp = float(commission_bp)
        self.half_spread_bp = float(half_spread_bp)
        self.adv_window = max(1, int(adv_window))
        self.min_adv_periods = max(1, int(min_adv_periods))
        self.require_liquidity_data = bool(require_liquidity_data)

    # ----------------------------------------------------------------
    # static helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _side_sign(side: str) -> int:
        if side == "BUY":
            return 1
        if side == "SELL":
            return -1
        raise ValueError(f"invalid side: {side}")

    @staticmethod
    def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
        required = {"code", "date", "open", "close", "turnover"}
        missing = required - set(prices.columns)
        if missing:
            raise ValueError(f"prices missing columns: {sorted(missing)}")

        x = prices.copy()
        x["code"] = x["code"].astype(str)
        x["date"] = pd.to_datetime(x["date"]).dt.normalize()

        for col in ["open", "close", "turnover"]:
            x[col] = pd.to_numeric(x[col], errors="coerce")

        return (
            x.sort_values(["code", "date"])
            .drop_duplicates(["code", "date"], keep="last")
            .reset_index(drop=True)
        )

    @staticmethod
    def _prepare_orders(orders: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "code", "side", "qty", "order_type", "holding_days"}
        missing = required - set(orders.columns)
        if missing:
            raise ValueError(f"orders missing columns: {sorted(missing)}")

        x = orders.copy()
        x["date"] = pd.to_datetime(x["date"]).dt.normalize()
        x["code"] = x["code"].astype(str)
        x["side"] = x["side"].astype(str).str.upper()
        x["order_type"] = x["order_type"].astype(str).str.upper()
        x["qty"] = pd.to_numeric(x["qty"], errors="coerce").fillna(0).astype(int)
        x["holding_days"] = pd.to_numeric(x["holding_days"], errors="coerce").fillna(0).astype(int)

        if "score" not in x.columns:
            x["score"] = 0.0
        x["score"] = pd.to_numeric(x["score"], errors="coerce").fillna(0.0)

        if "name" not in x.columns:
            x["name"] = ""

        if "shortable" not in x.columns:
            x["shortable"] = False
        x["shortable"] = x["shortable"].fillna(False).astype(bool)

        return x.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)

    # ----------------------------------------------------------------
    # execution helpers
    # ----------------------------------------------------------------

    def _adv_for_order(self, history: dict, code: str, date: pd.Timestamp) -> float:
        """Return ADV for a code on a given date using price history."""
        px_flat = []
        for c, frame in history.items():
            f = frame.reset_index()
            f["code"] = c
            px_flat.append(f)
        if not px_flat:
            return float("nan")
        combined = pd.concat(px_flat, ignore_index=True)
        return stock_adv_before(
            combined,
            date,
            code,
            window=self.adv_window,
            min_periods=self.min_adv_periods,
            strictly_before=True,
        )

    def _execution_price(
        self,
        *,
        base_price: float,
        qty: int,
        adv: float,
        direction: int,
    ) -> tuple[float, float]:
        if base_price <= 0:
            raise ValueError(f"base_price must be > 0: {base_price}")
        if qty <= 0:
            raise ValueError(f"qty must be > 0: {qty}")

        order_value = base_price * qty

        if self._is_valid_adv(adv) and adv > 0:
            impact_bp = self.impact_k_bp * float(np.sqrt(order_value / adv))
        else:
            impact_bp = 0.0

        total_bp = impact_bp + self.half_spread_bp + self.commission_bp
        execution_price = base_price * (1.0 + direction * total_bp / 10_000.0)

        return float(execution_price), float(total_bp)

    @staticmethod
    def _is_valid_adv(adv: float | None) -> bool:
        if adv is None:
            return False
        try:
            value = float(adv)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(value) and value > 0)

    @staticmethod
    def _planned_exit_date(
        trading_dates: list[pd.Timestamp],
        entry_date: pd.Timestamp,
        holding_days: int,
    ) -> pd.Timestamp | None:
        try:
            index = trading_dates.index(entry_date)
        except ValueError:
            return None

        exit_index = index + holding_days
        if exit_index >= len(trading_dates):
            return None

        return trading_dates[exit_index]

    @staticmethod
    def _mark_to_market_price(
        history: dict[str, pd.DataFrame],
        code: str,
        date: pd.Timestamp,
        fallback: float,
    ) -> float:
        frame = history.get(code)
        if frame is None or frame.empty:
            return fallback

        past = frame.loc[frame.index <= date]
        if past.empty:
            return fallback

        last_row = past.iloc[-1]
        if "close" not in last_row.index:
            return fallback

        value = float(last_row["close"])
        if np.isfinite(value) and value > 0:
            return value
        return fallback

    @staticmethod
    def _mark_at_open_price(
        history: dict[str, pd.DataFrame],
        code: str,
        date: pd.Timestamp,
        fallback: float,
    ) -> float:
        """寄付き時点の評価価格を返す。

        当日の有効なopenがあれば使用する。
        当日openが欠損している場合は、前営業日までの直近closeを使用する。
        当日closeは寄付き時点で未確定なので使用しない。
        """
        frame = history.get(code)
        if frame is None or frame.empty:
            return fallback

        if date in frame.index:
            row = frame.loc[date]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            if "open" in row.index:
                open_price = float(row["open"])
                if np.isfinite(open_price) and open_price > 0:
                    return open_price

        previous = frame.loc[frame.index < date]
        if previous.empty:
            return fallback

        prev_row = previous.iloc[-1]
        if "close" in prev_row.index:
            close_price = float(prev_row["close"])
            if np.isfinite(close_price) and close_price > 0:
                return close_price

        return fallback

    def _exposure_at_open(
        self,
        positions: list[Position],
        history: dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> dict[str, float]:
        """寄付き時点の既存ポジションexposureを計算する。

        当日引けで決済予定のポジションも、当日寄付き時点では
        まだ存在するため計算対象に含める。
        """
        long_value = 0.0
        short_value = 0.0

        for position in positions:
            price = self._mark_at_open_price(
                history,
                position.code,
                date,
                position.entry_price,
            )
            value = price * position.qty

            if position.side == "BUY":
                long_value += value
            elif position.side == "SELL":
                short_value += value
            else:
                raise ValueError(f"invalid position side: {position.side}")

        return {
            "long": float(long_value),
            "short": float(short_value),
            "gross": float(long_value + short_value),
            "net": float(long_value - short_value),
        }

    def _exposure(
        self,
        positions: list[Position],
        history: dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> dict[str, float]:
        long_value = 0.0
        short_value = 0.0

        for position in positions:
            price = self._mark_to_market_price(
                history,
                position.code,
                date,
                position.entry_price,
            )
            value = price * position.qty

            if position.side == "BUY":
                long_value += value
            else:
                short_value += value

        return {
            "long": float(long_value),
            "short": float(short_value),
            "gross": float(long_value + short_value),
            "net": float(long_value - short_value),
        }

    # ----------------------------------------------------------------
    # order selection
    # ----------------------------------------------------------------

    def _select_orders(
        self,
        candidates: list[dict],
        base: dict[str, float],
    ) -> tuple[list[dict], list[tuple[dict, str]]]:
        """既存建玉を含めて注文を選択する。

        gross/long/shortを先に判定し、netはlong-shortの
        バッチ全体で判定する。
        """
        selected: list[dict] = []
        rejected: list[tuple[dict, str]] = []

        long_value = base["long"]
        short_value = base["short"]

        sorted_candidates = sorted(
            candidates,
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )

        for candidate in sorted_candidates:
            if len(selected) >= self.risk.max_orders_per_day:
                rejected.append((candidate, "DAILY_ORDER_LIMIT"))
                continue

            value = float(candidate["entry_value"])

            if value > self.risk.max_single_name_exposure_yen:
                rejected.append((candidate, "SINGLE_NAME_LIMIT"))
                continue

            next_long = long_value
            next_short = short_value

            if candidate["side"] == "BUY":
                next_long += value
            else:
                next_short += value

            if next_long > self.risk.max_long_exposure_yen:
                rejected.append((candidate, "LONG_LIMIT"))
                continue
            if next_short > self.risk.max_short_exposure_yen:
                rejected.append((candidate, "SHORT_LIMIT"))
                continue
            if next_long + next_short > self.risk.max_gross_exposure_yen:
                rejected.append((candidate, "GROSS_LIMIT"))
                continue

            selected.append(candidate)
            long_value = next_long
            short_value = next_short

        def selected_exposure() -> tuple[float, float, float]:
            sel_long = base["long"] + sum(
                float(c["entry_value"]) for c in selected if c["side"] == "BUY"
            )
            sel_short = base["short"] + sum(
                float(c["entry_value"]) for c in selected if c["side"] == "SELL"
            )
            return sel_long, sel_short, sel_long - sel_short

        # net exposureを満たすまで弱い注文から削除
        while selected:
            _, _, net_value = selected_exposure()

            if abs(net_value) <= self.risk.max_net_exposure_yen:
                break

            excessive_side = "BUY" if net_value > 0 else "SELL"
            removable = [c for c in selected if c["side"] == excessive_side]

            if not removable:
                break

            weakest = min(
                removable,
                key=lambda x: float(x.get("score", 0.0)),
            )
            selected.remove(weakest)
            rejected.append((weakest, "NET_LIMIT"))

        _, _, net_value = selected_exposure()
        net_within_limit = abs(net_value) <= self.risk.max_net_exposure_yen

        if not net_within_limit:
            return [], rejected

        # require_both_sides
        if self.risk.require_both_sides:
            sides = {c["side"] for c in selected}
            if not {"BUY", "SELL"}.issubset(sides):
                return [], rejected

        return selected, rejected

    # ----------------------------------------------------------------
    # carry cost
    # ----------------------------------------------------------------

    @staticmethod
    def _accrue_carry(
        positions: list[Position],
        current_date: pd.Timestamp,
        daily_borrow_rate: float,
        daily_lend_rate: float,
    ) -> list[Position]:
        """未決済ポジションのcarryを暦日単位で発生させる。

        carryはここではcashから控除せず、Position.accrued_carryに
        未払費用として蓄積する。

        NAVでは未払carryを控除し、決済時にcashから支払う。
        """
        updated: list[Position] = []
        for pos in positions:
            if pos.last_carry_date >= current_date:
                updated.append(pos)
                continue

            cal_days = (current_date - pos.last_carry_date).days
            if cal_days <= 0:
                updated.append(pos)
                continue

            rate = daily_lend_rate if pos.side == "SELL" else daily_borrow_rate
            carry = pos.entry_price * rate * cal_days * pos.qty

            pos.accrued_carry += carry
            pos.accrued_carry_days += cal_days
            pos.last_carry_date = current_date
            updated.append(pos)

        return updated

    # ----------------------------------------------------------------
    # main run
    # ----------------------------------------------------------------

    def run(
        self,
        orders: pd.DataFrame,
        prices: pd.DataFrame,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> PortfolioResult:
        if orders is None or orders.empty:
            empty = pd.DataFrame()
            return PortfolioResult(
                trades=empty,
                rejected_orders=empty,
                daily_ledger=empty,
                open_positions=empty,
            )

        px = self._prepare_prices(prices)
        od = self._prepare_orders(orders)

        price_map = {}
        for _, row in px.iterrows():
            key = (str(row["code"]), pd.Timestamp(row["date"]).normalize())
            price_map[key] = row.to_dict()
        history = {
            str(code): group.set_index("date").sort_index()
            for code, group in px.groupby("code", sort=False)
        }

        trading_dates = sorted(pd.Timestamp(date) for date in px["date"].unique())

        if start_date is not None:
            start_ts = pd.Timestamp(start_date).normalize()
            trading_dates = [d for d in trading_dates if d >= start_ts]

        if end_date is not None:
            end_ts = pd.Timestamp(end_date).normalize()
            trading_dates = [d for d in trading_dates if d <= end_ts]

        if not trading_dates:
            empty = pd.DataFrame()
            return PortfolioResult(
                trades=empty,
                rejected_orders=empty,
                daily_ledger=empty,
                open_positions=empty,
            )

        # state
        cash = self.initial_capital
        positions: list[Position] = []
        trades: list[dict] = []
        rejected_orders: list[dict] = []
        daily_ledger: list[dict] = []

        for date in trading_dates:
            # --- 1. accrue carry on existing positions ---
            positions = self._accrue_carry(
                positions,
                date,
                self.daily_interest,
                self.daily_lending,
            )

            # --- 1b. save positions before closing for open-time risk check ---
            pre_close_positions = positions[:]

            # --- 2. close positions whose planned_exit_date <= date ---
            closing_positions: list[Position] = []
            remaining_positions: list[Position] = []

            for pos in positions:
                if pos.planned_exit_date <= date:
                    closing_positions.append(pos)
                else:
                    remaining_positions.append(pos)

            positions = remaining_positions

            deferred_positions: list[Position] = []
            settled_positions: list[Position] = []

            for pos in closing_positions:
                # 決済日の価格を取得。欠損している場合は決済を延期
                frame_px = history.get(str(pos.code))
                if frame_px is None or frame_px.empty:
                    deferred_positions.append(pos)
                    continue

                exact_available = (
                    date in frame_px.index
                    and np.isfinite(frame_px.loc[date, "close"])
                    and frame_px.loc[date, "close"] > 0
                )
                if not exact_available:
                    deferred_positions.append(pos)
                    continue

                exit_price_raw = float(frame_px.loc[date, "close"])

                exit_adv = self._adv_for_order(history, pos.code, date)
                if self.require_liquidity_data and not self._is_valid_adv(exit_adv):
                    deferred_positions.append(pos)
                    continue

                direction = -self._side_sign(pos.side)
                exit_price, exit_slippage_bp = self._execution_price(
                    base_price=exit_price_raw,
                    qty=pos.qty,
                    adv=exit_adv,
                    direction=direction,
                )

                gross_pnl = (exit_price - pos.entry_price) * self._side_sign(pos.side) * pos.qty
                total_carry = pos.accrued_carry
                net_pnl = gross_pnl - total_carry

                # cash更新: 決済
                if pos.side == "BUY":
                    cash += exit_price * pos.qty
                else:
                    cash -= exit_price * pos.qty

                # carryをcashから支払う
                cash -= total_carry

                settled_positions.append(pos)

                trades.append(
                    {
                        "position_id": pos.position_id,
                        "code": pos.code,
                        "name": pos.name,
                        "side": pos.side,
                        "qty": pos.qty,
                        "entry_date": str(pos.entry_date.date()),
                        "exit_date": str(date.date()),
                        "planned_exit_date": str(pos.planned_exit_date.date()),
                        "entry": pos.entry_price,
                        "exit": exit_price,
                        "entry_adv": pos.entry_adv,
                        "exit_adv": exit_adv,
                        "entry_slippage_bp": pos.entry_slippage_bp,
                        "exit_slippage_bp": exit_slippage_bp,
                        "carry_cost": total_carry,
                        "carry_days": pos.accrued_carry_days,
                        "gross_pnl": gross_pnl,
                        "pnl": net_pnl,
                        "score": pos.score,
                    }
                )

            positions = remaining_positions + deferred_positions

            for dpos in deferred_positions:
                rejected_orders.append(
                    {
                        "code": dpos.code,
                        "name": dpos.name,
                        "side": dpos.side,
                        "qty": dpos.qty,
                        "rejection_date": str(date.date()),
                        "reason": "NO_EXIT_PRICE_DEFERRED",
                    }
                )

            # --- 3. process new orders for this date ---
            day_orders = od[od["date"] == date]
            candidates: list[dict] = []

            # 寄付き時点では決済予定のポジションも存在するため既存コードとして扱う
            existing_codes: set[str] = {p.code for p in pre_close_positions}
            seen_codes_today: set[str] = set()

            for _, row in day_orders.iterrows():
                code = str(row["code"])
                side = str(row["side"]).upper()
                qty = int(row["qty"])
                holding_days = int(row["holding_days"])
                order_type = str(row.get("order_type", "")).upper()

                reject = None

                if order_type not in {"MKT_OPEN", ""}:
                    reject = "UNSUPPORTED_ORDER_TYPE"
                elif code in existing_codes:
                    reject = "EXISTING_POSITION"
                elif code in seen_codes_today:
                    reject = "DUPLICATE_CODE_SAME_DAY"
                elif side not in {"BUY", "SELL"}:
                    reject = "INVALID_SIDE"
                elif qty <= 0:
                    reject = "INVALID_QTY"
                elif holding_days < 1:
                    reject = "INVALID_HOLDING_DAYS"
                elif side == "SELL" and not self.risk.allow_short_without_confirmed_shortability:
                    shortable = bool(row.get("shortable", False))
                    if not shortable:
                        reject = "NOT_SHORTABLE"

                if reject is not None:
                    rejected_orders.append(
                        {
                            **row.to_dict(),
                            "rejection_date": str(date.date()),
                            "reason": reject,
                        }
                    )
                    continue

                key = (code, date)
                price_row = price_map.get(key)
                if price_row is None:
                    rejected_orders.append(
                        {
                            **row.to_dict(),
                            "rejection_date": str(date.date()),
                            "reason": "NO_PRICE_DATA",
                        }
                    )
                    continue

                base_price = float(price_row["open"])
                if not np.isfinite(base_price) or base_price <= 0:
                    rejected_orders.append(
                        {
                            **row.to_dict(),
                            "rejection_date": str(date.date()),
                            "reason": "INVALID_OPEN_PRICE",
                        }
                    )
                    continue

                adv = self._adv_for_order(history, code, date)

                if self.require_liquidity_data and not self._is_valid_adv(adv):
                    rejected_orders.append(
                        {
                            **row.to_dict(),
                            "rejection_date": str(date.date()),
                            "reason": "LIQUIDITY_DATA_UNAVAILABLE",
                        }
                    )
                    continue

                direction = self._side_sign(side)
                entry_price, entry_slippage_bp = self._execution_price(
                    base_price=base_price,
                    qty=qty,
                    adv=adv if self._is_valid_adv(adv) else 0.0,
                    direction=direction,
                )

                entry_value = entry_price * qty

                exit_date = self._planned_exit_date(trading_dates, date, holding_days)
                if exit_date is None:
                    rejected_orders.append(
                        {
                            **row.to_dict(),
                            "rejection_date": str(date.date()),
                            "reason": "NO_EXIT_DATE_WITHIN_TEST_WINDOW",
                        }
                    )
                    continue

                seen_codes_today.add(code)

                candidates.append(
                    {
                        "code": code,
                        "name": str(row.get("name", "")),
                        "side": side,
                        "qty": qty,
                        "entry_price": entry_price,
                        "entry_value": entry_value,
                        "adv": adv,
                        "entry_slippage_bp": entry_slippage_bp,
                        "score": float(row.get("score", 0.0)),
                        "holding_days": holding_days,
                        "exit_date": exit_date,
                    }
                )

            # 寄付き時点のリスク判定: 当日引けで決済予定のポジションも含める
            base_exposure = self._exposure_at_open(pre_close_positions, history, date)
            selected, day_rejected = self._select_orders(candidates, base_exposure)

            for candidate, reason in day_rejected:
                rejected_orders.append(
                    {
                        "code": candidate["code"],
                        "name": candidate.get("name", ""),
                        "side": candidate["side"],
                        "qty": candidate["qty"],
                        "entry_price": candidate["entry_price"],
                        "score": candidate["score"],
                        "rejection_date": str(date.date()),
                        "reason": reason,
                    }
                )

            # --- 4. open new positions ---
            for cand in selected:
                pos = Position(
                    position_id=uuid.uuid4().hex[:12],
                    code=cand["code"],
                    name=cand["name"],
                    side=cand["side"],
                    qty=cand["qty"],
                    entry_date=date,
                    planned_exit_date=cand["exit_date"],
                    entry_price=cand["entry_price"],
                    entry_adv=cand["adv"],
                    entry_slippage_bp=cand["entry_slippage_bp"],
                    score=cand["score"],
                    last_carry_date=date,
                )

                if cand["side"] == "BUY":
                    cash -= cand["entry_price"] * cand["qty"]
                else:
                    cash += cand["entry_price"] * cand["qty"]
                positions.append(pos)

            # --- 5. record daily ledger ---
            exposure = self._exposure(positions, history, date)
            total_accrued_carry = sum(p.accrued_carry for p in positions)
            nav = cash + exposure["long"] - exposure["short"] - total_accrued_carry

            daily_ledger.append(
                {
                    "date": str(date.date()),
                    "cash": cash,
                    "long_exposure": exposure["long"],
                    "short_exposure": exposure["short"],
                    "gross_exposure": exposure["gross"],
                    "net_exposure": exposure["net"],
                    "open_position_count": len(positions),
                    "accrued_carry": total_accrued_carry,
                    "nav": nav,
                }
            )

        result = PortfolioResult(
            trades=pd.DataFrame(trades) if trades else pd.DataFrame(),
            rejected_orders=pd.DataFrame(rejected_orders) if rejected_orders else pd.DataFrame(),
            daily_ledger=pd.DataFrame(daily_ledger) if daily_ledger else pd.DataFrame(),
            open_positions=(
                pd.DataFrame([asdict(p) for p in positions]) if positions else pd.DataFrame()
            ),
        )

        return result
