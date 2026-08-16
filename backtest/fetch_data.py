"""
바이낸스 공개 API에서 BTCUSDT 1시간봉 캔들을 최근 3년치 받아 CSV로 저장한다.
공개 klines 엔드포인트라 API 키가 필요 없다.
"""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
YEARS = 3
LIMIT = 1000  # 바이낸스 klines 1회 호출 최대 개수
BASE_URL = "https://api.binance.com/api/v3/klines"

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / f"{SYMBOL}_{INTERVAL}_{YEARS}y.csv"

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_klines(start_ms: int, end_ms: int) -> list:
    all_rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": LIMIT,
        }
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
            raise RuntimeError("바이낸스 API 요청이 반복적으로 실패했습니다.")

        if not rows:
            break

        all_rows.extend(rows)
        last_open_time = rows[-1][0]
        cursor = last_open_time + 1
        print(f"  {datetime.fromtimestamp(last_open_time / 1000, tz=timezone.utc)} 까지 수집 ({len(all_rows)}개)")
        time.sleep(0.25)  # 레이트리밋 여유

    return all_rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * YEARS)
    print(f"{SYMBOL} {INTERVAL} 캔들 수집: {start.date()} ~ {end.date()}")

    rows = fetch_klines(int(start.timestamp() * 1000), int(end.timestamp() * 1000))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time", "trades"]]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"저장 완료: {OUTPUT_FILE} ({len(df)}행)")


if __name__ == "__main__":
    main()
