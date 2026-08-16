"""
비트코인 변동성의 3대 요인을 백테스트에 반영하기 위한 시장 컨텍스트 피처.

1. 레버리지 청산 캐스케이드 -> 펀딩비율(funding rate)로 포지션 쏠림 감지
2. 거시경제/유동성 환경 -> FOMC 회의 발표일 전후는 신규 진입 회피
3. 얕은 유동성/뉴스 충격 -> 변동성 급등 구간(과거 90일 상위 10%) 신규 진입 회피
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
FUNDING_FILE = DATA_DIR / "BTCUSDT_funding_3y.csv"

# 펀딩비율 분포(3년치) 기준: 90~95분위 근방을 "롱 과열", 10분위 근방을 "숏 과열"로 설정
FUNDING_HIGH_THRESHOLD = 0.00015   # 이보다 높으면 롱 쏠림 -> 신규 롱 진입 회피
FUNDING_LOW_THRESHOLD = -0.00005   # 이보다 낮으면 숏 쏠림 -> 신규 숏 진입 회피

# 미 연준 FOMC 회의 발표일(정책 발표가 나오는 둘째 날). federalreserve.gov 공개 일정 기준.
FOMC_DATES = [
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29",
]

VOL_WINDOW_HOURS = 24        # 실현 변동성 계산 구간
VOL_LOOKBACK_BARS = 24 * 90  # 상대적 고변동성 판단을 위한 과거 90일 창
VOL_PERCENTILE = 0.90        # 상위 10%를 고변동성(뉴스/유동성 충격 위험) 구간으로 간주


def load_funding() -> pd.DataFrame:
    df = pd.read_csv(FUNDING_FILE)
    df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True, format="ISO8601")
    # 원본에 밀리초 단위 지터가 섞여 있어(예: 08:00:00.001) 정시로 정규화
    df["funding_time"] = df["funding_time"].dt.floor("h")
    return df.sort_values("funding_time")


def add_market_context(price_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)

    # --- 1. 펀딩비율: 8시간 데이터를 1시간봉에 맞춰 forward-fill로 병합 ---
    funding = load_funding()
    out = pd.merge_asof(
        out.sort_values("open_time"),
        funding.rename(columns={"funding_time": "open_time", "funding_rate": "funding_rate"}),
        on="open_time",
        direction="backward",
    )
    out["funding_rate"] = out["funding_rate"].fillna(0.0)
    # 실제 펀딩 정산이 발생하는 시점(그 시각의 봉)만 True로 표시 (비용 이중 적용 방지)
    out["is_funding_bar"] = out["open_time"].isin(funding["funding_time"])

    # --- 2. FOMC 발표일 전후 1일은 매크로 리스크 구간으로 표시 ---
    fomc_ts = pd.to_datetime(FOMC_DATES, utc=True)
    out["date_only"] = out["open_time"].dt.floor("D")
    macro_days = set()
    for d in fomc_ts:
        for delta in (-1, 0, 1):
            macro_days.add((d + pd.Timedelta(days=delta)).floor("D"))
    out["is_macro_window"] = out["date_only"].isin(macro_days)
    out = out.drop(columns=["date_only"])

    # --- 3. 변동성 급등(과거 90일 대비 상위 10%) 구간 표시 ---
    log_ret = np.log(out["close"] / out["close"].shift(1))
    realized_vol = log_ret.rolling(VOL_WINDOW_HOURS, min_periods=VOL_WINDOW_HOURS).std()
    vol_threshold = realized_vol.rolling(VOL_LOOKBACK_BARS, min_periods=VOL_WINDOW_HOURS * 5).quantile(VOL_PERCENTILE)
    out["realized_vol"] = realized_vol
    out["is_high_vol_regime"] = (realized_vol > vol_threshold.shift(1)).fillna(False)

    out["allow_long_entry"] = (
        (out["funding_rate"] <= FUNDING_HIGH_THRESHOLD)
        & (~out["is_macro_window"])
        & (~out["is_high_vol_regime"])
    )
    out["allow_short_entry"] = (
        (out["funding_rate"] >= FUNDING_LOW_THRESHOLD)
        & (~out["is_macro_window"])
        & (~out["is_high_vol_regime"])
    )

    return out
