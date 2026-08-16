"""
바이낸스 선물(perpetual) 공개 API에서 BTCUSDT 펀딩비율을 3년치 받아 CSV로 저장한다.
펀딩비율은 레버리지 포지션 쏠림(청산 캐스케이드 위험)의 대리 지표로 사용한다.
8시간마다 발생하므로, 이후 1시간봉 데이터와 합칠 때는 forward-fill로 정렬한다.
"""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
YEARS = 3
LIMIT = 1000
BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / f"{SYMBOL}_funding_{YEARS}y.csv"


def fetch_funding(start_ms: int, end_ms: int) -> list:
    all_rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": SYMBOL, "startTime": cursor, "endTime": end_ms, "limit": LIMIT}
        for attempt in range(5):
            try:
                res = requests.get(BASE_URL, params=params, timeout=15)
                res.raise_for_status()
                rows = res.json()
                break
            except requests.RequestException as exc:
                wait = 2 ** attempt
                print(f"  요청 실패({exc}), {wait}s 후 재시도")
                time.sleep(wait)
        else:
            raise RuntimeError("바이낸스 펀딩비율 API 요청이 반복적으로 실패했습니다.")

        if not rows:
            break

        all_rows.extend(rows)
        cursor = rows[-1]["fundingTime"] + 1
        print(f"  {datetime.fromtimestamp(rows[-1]['fundingTime'] / 1000, tz=timezone.utc)} 까지 수집 ({len(all_rows)}개)")
        time.sleep(0.25)

    return all_rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * YEARS)
    print(f"{SYMBOL} 펀딩비율 수집: {start.date()} ~ {end.date()}")

    rows = fetch_funding(int(start.timestamp() * 1000), int(end.timestamp() * 1000))

    df = pd.DataFrame(rows)[["fundingTime", "fundingRate"]]
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.rename(columns={"fundingTime": "funding_time", "fundingRate": "funding_rate"})
    df = df.drop_duplicates(subset="funding_time").sort_values("funding_time").reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"저장 완료: {OUTPUT_FILE} ({len(df)}행)")
    print(df["funding_rate"].describe())


if __name__ == "__main__":
    main()
