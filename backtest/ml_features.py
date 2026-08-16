"""
머신러닝 신호 생성을 위한 피처/라벨 엔지니어링.

모든 피처는 시점 t까지의 정보만 사용한다(미래참조 금지).
라벨(미래 수익률)만 미래 시점 데이터를 쓰며, 학습에만 사용하고 피처로는 쓰지 않는다.
"""
import numpy as np
import pandas as pd

from indicators import sma, ema, rsi, macd
from market_context import add_market_context

FEATURE_COLUMNS = [
    "ret_1h", "ret_4h", "ret_24h", "ret_168h",
    "rsi_14",
    "macd_hist_norm",
    "sma_ratio_20_60",
    "price_to_sma20",
    "bb_width", "bb_position",
    "realized_vol_24h",
    "volume_ratio",
    "funding_rate", "funding_rate_change",
    "is_macro_window", "is_high_vol_regime",
    "sentiment_score",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
]


def build_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = add_market_context(price_df.copy())
    close = df["close"]

    df["ret_1h"] = close.pct_change(1)
    df["ret_4h"] = close.pct_change(4)
    df["ret_24h"] = close.pct_change(24)
    df["ret_168h"] = close.pct_change(168)

    df["rsi_14"] = rsi(close, 14)

    macd_line, signal_line, hist = macd(close)
    df["macd_hist_norm"] = hist / close  # 가격 스케일 영향 제거

    sma20 = sma(close, 20)
    sma60 = sma(close, 60)
    df["sma_ratio_20_60"] = (sma20 / sma60) - 1
    df["price_to_sma20"] = (close / sma20) - 1

    std20 = close.rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["bb_width"] = (bb_upper - bb_lower) / sma20
    df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)

    log_ret = np.log(close / close.shift(1))
    df["realized_vol_24h"] = log_ret.rolling(24, min_periods=24).std()

    vol_ma = df["volume"].rolling(24, min_periods=24).mean()
    df["volume_ratio"] = df["volume"] / vol_ma

    df["funding_rate_change"] = df["funding_rate"].diff().fillna(0.0)
    df["is_macro_window"] = df["is_macro_window"].astype(float)
    df["is_high_vol_regime"] = df["is_high_vol_regime"].astype(float)

    hour = df["open_time"].dt.hour
    dow = df["open_time"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


def build_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    """N봉 후 수익률. 라벨 계산에만 미래 데이터를 쓰며 피처에는 사용하지 않는다."""
    return (df["close"].shift(-horizon) / df["close"]) - 1
