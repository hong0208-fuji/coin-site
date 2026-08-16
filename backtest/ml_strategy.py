"""
ML 예측 수익률을 매매 신호로 변환한다.

진입: 예측 수익률이 임계값을 넘으면 롱/숏 진입 후보. 기존 3요인 필터
(펀딩비율 쏠림/FOMC/변동성 급등, market_context.py)를 그대로 게이트로 사용.
청산: 예측에 사용한 예측 구간(horizon)만큼 지난 시점에 시간 기반 청산 신호를 발생시키고,
      손절/익절은 backtest.py가 매 봉마다 별도로 체크한다.
"""
import pandas as pd

DEFAULT_PARAMS = {
    "horizon": 4,             # 예측 구간(봉 수, 1시간봉 기준 4시간)
    "entry_threshold_long": 0.004,
    "entry_threshold_short": 0.004,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04,
    "max_holding_bars": 24,   # 시간기반 청산이 안 걸릴 경우의 안전장치
}


def generate_ml_signals(df: pd.DataFrame, predictions, params: dict):
    p = {**DEFAULT_PARAMS, **params}
    out = df.copy()
    out["predicted_return"] = predictions

    raw_long = (out["predicted_return"] > p["entry_threshold_long"]) & out["allow_long_entry"]
    raw_short = (out["predicted_return"] < -p["entry_threshold_short"]) & out["allow_short_entry"]

    out["buy_signal"] = raw_long.fillna(False)
    out["short_signal"] = raw_short.fillna(False)

    # 진입 후 horizon봉 뒤에 시간기반 청산 신호 발생 (해당 시점에 열려있는 포지션이면 청산)
    out["sell_signal"] = out["buy_signal"].shift(p["horizon"], fill_value=False)
    out["cover_signal"] = out["short_signal"].shift(p["horizon"], fill_value=False)

    return out, p
