"""シグナル生成モデル（FR-MODEL-01/02/04）。

SignalModel インターフェースを介してルールベース/ML を差し替え可能にする。
指数方向は予測せず、相対的な上位/下位を狙う設計。

look-ahead 回避:
  当日終値は as_of 時点で未確定のため使わない。
  前営業日までの価格のみ使用する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar import previous_business_day


class SignalModel(ABC):
    """シグナル生成インターフェース。"""

    @abstractmethod
    def generate(self, prices: pd.DataFrame, as_of: str) -> pd.DataFrame:
        """returns: code, side('BUY'|'SELL'), score, limit_price(optional)"""
        raise NotImplementedError


class MeanReversionRule(SignalModel):
    """ルールベースの例（動作確認用ダミー）。収益性は担保しない。"""

    def __init__(self, lookback: int = 5, top_n: int = 5):
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        self.lookback = lookback
        self.top_n = top_n

    def generate(self, prices: pd.DataFrame, as_of: str) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["code", "side", "score", "limit_price"])

        if prices is None or prices.empty:
            return empty

        required = {"code", "date"}
        missing = required - set(prices.columns)
        if missing:
            raise ValueError(f"prices missing columns: {sorted(missing)}")

        as_of_d = pd.Timestamp(as_of).date()
        cutoff = pd.Timestamp(previous_business_day(as_of_d))

        df = prices.copy()
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        ).dt.normalize()
        df["code"] = df["code"].astype(str).str.strip()

        df = df.dropna(subset=["date"])
        df = df[df["date"] <= cutoff]

        if df.empty:
            return empty

        df = df.sort_values(["code", "date"])

        price_col = "adj_close" if "adj_close" in df.columns else "close"

        rets: dict[str, float] = {}
        for code, g in df.groupby("code"):
            g = g.tail(self.lookback + 1)
            if len(g) < self.lookback + 1:
                continue
            first = g[price_col].iloc[0]
            last = g[price_col].iloc[-1]
            if first is not None and first > 0 and np.isfinite(first) and np.isfinite(last):
                rets[str(code)] = (last / first) - 1.0

        if len(rets) < 2:
            return empty

        ser = pd.Series(rets, dtype=float).dropna().sort_values()

        n = min(
            self.top_n,
            len(ser) // 2,
        )

        if n < 1:
            return empty

        buys = ser.iloc[:n]
        sells = ser.iloc[-n:]

        overlap = set(buys.index) & set(sells.index)
        if overlap:
            raise RuntimeError(f"BUY/SELL overlap detected: {sorted(overlap)}")

        rows: list[dict] = []
        for code, r in buys.items():
            rows.append(
                {
                    "code": code,
                    "side": "BUY",
                    "score": float(-r),
                    "limit_price": np.nan,
                }
            )
        for code, r in sells.items():
            rows.append(
                {
                    "code": code,
                    "side": "SELL",
                    "score": float(r),
                    "limit_price": np.nan,
                }
            )

        return pd.DataFrame(rows, columns=["code", "side", "score", "limit_price"])


@dataclass
class MeanReversionConfig:
    """VolatilityAdjustedMeanReversion の設定。"""
    lookback: int = 20
    z_entry: float = 0.5
    z_exit: float = 0.0
    vol_lookback: int = 60
    min_vol_percentile: float = 5.0
    max_vol_percentile: float = 95.0
    min_adv_multiple: float = 0.5
    max_adv_multiple: float = 3.0
    min_vol_target: float = 0.10
    max_vol_target: float = 0.40
    smoothing_periods: int = 5
    use_sample_weights: bool = True
    top_n: int = 10

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError(f"lookback must be >= 1: {self.lookback}")
        if self.top_n < 1:
            raise ValueError(f"top_n must be >= 1: {self.top_n}")
        if self.vol_lookback < 1:
            raise ValueError(f"vol_lookback must be >= 1: {self.vol_lookback}")
        if not 0 <= self.min_vol_percentile < self.max_vol_percentile <= 100:
            raise ValueError(
                f"vol_percentile range invalid: "
                f"min={self.min_vol_percentile} max={self.max_vol_percentile}"
            )


class VolatilityAdjustedMeanReversion(SignalModel):
    """ボラティリティ調整済み平均回帰シグナル（PR-1）。

    z-score で標準化したリターンに対し、ボラティリティに応じて
    シグナル強度を調整する。高ボラ銘柄のシグナルを抑制し、
    低ボラ銘柄のシグナルを増強する。
    """

    def __init__(self, config: MeanReversionConfig | dict | None = None):
        if config is None:
            config = MeanReversionConfig()
        if isinstance(config, dict):
            config = MeanReversionConfig(**config)
        self.config = config

    def generate(self, prices: pd.DataFrame, as_of: str) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["code", "side", "score", "limit_price"])

        if prices is None or prices.empty:
            return empty

        required = {"code", "date", "close"}
        missing = required - set(prices.columns)
        if missing:
            raise ValueError(f"prices missing columns: {sorted(missing)}")

        cfg = self.config
        as_of_d = pd.Timestamp(as_of).date()
        cutoff = pd.Timestamp(previous_business_day(as_of_d))

        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["code"] = df["code"].astype(str).str.strip()
        df = df.dropna(subset=["date"])
        df = df[df["date"] <= cutoff]

        if df.empty:
            return empty

        df = df.sort_values(["code", "date"])
        price_col = "adj_close" if "adj_close" in df.columns else "close"

        results: list[dict] = []

        for code, g in df.groupby("code"):
            g = g.sort_values("date").reset_index(drop=True)
            if len(g) < max(cfg.lookback + 1, cfg.vol_lookback + 1):
                continue

            prices_series = g[price_col].values.astype(float)
            if not np.all(np.isfinite(prices_series)):
                continue

            rets = np.diff(prices_series) / prices_series[:-1]
            if len(rets) < cfg.lookback:
                continue

            recent_rets = rets[-cfg.lookback:]
            mean_ret = np.mean(recent_rets)
            std_ret = np.std(recent_rets)
            if std_ret < 1e-10:
                continue

            vol_series = pd.Series(rets).rolling(cfg.vol_lookback).std().values
            current_vol = vol_series[-1] if len(vol_series) > 0 and np.isfinite(vol_series[-1]) else std_ret
            if not np.isfinite(current_vol) or current_vol < 1e-10:
                continue

            z = (recent_rets[-1] - mean_ret) / std_ret

            vol_percentile = np.mean(vol_series[-cfg.vol_lookback:] <= current_vol) * 100 if len(vol_series) > cfg.vol_lookback else 50.0
            vol_multiplier = np.clip(
                1.0 - (vol_percentile - cfg.min_vol_percentile) / (cfg.max_vol_percentile - cfg.min_vol_percentile),
                0.0,
                1.0,
            )

            score = -z * (0.5 + 0.5 * vol_multiplier)

            if score > cfg.z_entry:
                results.append({
                    "code": code,
                    "side": "BUY",
                    "score": float(score),
                    "limit_price": np.nan,
                })
            elif score < -cfg.z_entry:
                results.append({
                    "code": code,
                    "side": "SELL",
                    "score": float(-score),
                    "limit_price": np.nan,
                })

        if not results:
            return empty

        result_df = pd.DataFrame(results, columns=["code", "side", "score", "limit_price"])
        result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)

        if len(result_df) > cfg.top_n * 2:
            buys = result_df[result_df["side"] == "BUY"].head(cfg.top_n)
            sells = result_df[result_df["side"] == "SELL"].head(cfg.top_n)
            return pd.concat([buys, sells], ignore_index=True).sort_values("score", ascending=False)

        return result_df


def model_from_config(model_cfg: dict) -> SignalModel:
    """設定 dict からモデルインスタンスを生成するファクトリ。

    model_cfg の type フィールドに基づいて適切なモデルを返す:
      - "mean_reversion" (デフォルト): MeanReversionRule
      - "volatility_adjusted"        : VolatilityAdjustedMeanReversion

    type 未指定時は MeanReversionRule を返す（後方互換）。
    """
    model_type = str(model_cfg.get("type", "mean_reversion")).strip().lower()

    if model_type == "volatility_adjusted":
        config = MeanReversionConfig(**{
            k: v for k, v in model_cfg.items()
            if k != "type"
        })
        return VolatilityAdjustedMeanReversion(config)

    # デフォルト: mean_reversion
    return MeanReversionRule(
        lookback=int(model_cfg.get("lookback", 5)),
        top_n=int(model_cfg.get("top_n", 5)),
    )
