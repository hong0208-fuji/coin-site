"""
SNS/시장 심리 감정 데이터를 alternative.me의 Crypto Fear & Greed Index에서 받아온다.

3년치 개별 SNS 게시글을 직접 수집해 감정분류하는 것은 이 프로젝트 범위에서는
비현실적이다(트위터/X API 유료+제한적, 3년치 원문 텍스트 NLP 처리 비용이 매우 큼).
대신 이 지수는 소셜미디어 여론/설문/모멘텀/변동성 등을 종합해 매일 계산되는
공개 무료 지표로, "군중 심리(긍정/중립/부정)"라는 요청 의도를 대체할 수 있는
실용적인 대안이다. 요청하신 3단계 점수(부정=0, 중립=1, 긍정=2)로 매핑한다.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.alternative.me/fng/"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / "sentiment_fng_3y.csv"

SCORE_MAP = {
    "Extreme Fear": 0,
    "Fear": 0,
    "Neutral": 1,
    "Greed": 2,
    "Extreme Greed": 2,
}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    res = requests.get(BASE_URL, params={"limit": 0, "format": "json"}, timeout=30)
    res.raise_for_status()
    data = res.json()["data"]

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.floor("D")
    df["value"] = df["value"].astype(int)
    df["sentiment_score"] = df["value_classification"].map(SCORE_MAP)
    df = df[["date", "value", "value_classification", "sentiment_score"]]
    df = df.rename(columns={"value": "fng_value", "value_classification": "fng_label"})
    df = df.sort_values("date").reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"저장 완료: {OUTPUT_FILE} ({len(df)}행, {df['date'].min().date()} ~ {df['date'].max().date()})")
    print(df["sentiment_score"].value_counts().sort_index())


if __name__ == "__main__":
    main()
