"""
파라미터화된 이동평균 크로스 + RSI 필터 전략.

매수: 단기 이동평균이 장기 이동평균을 상향 돌파(골든크로스)하고,
      RSI가 과매수 기준보다 낮을 때 (이미 과열된 상태에서 진입하지 않도록)
매도(지표 신호): 단기 이동평균이 장기 이동평균을 하향 돌파(데드크로스)하거나
                RSI가 과매수 기준을 넘어설 때
손절/익절은 backtest.py에서 진입가 대비 등락률로 별도 체크한다.
"""
from indicators import add_indicators

DEFAULT_PARAMS = {
    "short_window": 20,
    "long_window": 60,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04,
}


def generate_signals(df, params: dict):
    p = {**DEFAULT_PARAMS, **params}
    out = add_indicators(df, p["short_window"], p["long_window"], p["rsi_period"])

    prev_short = out["sma_short"].shift(1)
    prev_long = out["sma_long"].shift(1)

    golden_cross = (prev_short <= prev_long) & (out["sma_short"] > out["sma_long"])
    dead_cross = (prev_short >= prev_long) & (out["sma_short"] < out["sma_long"])

    out["buy_signal"] = golden_cross & (out["rsi"] < p["rsi_overbought"])
    out["sell_signal"] = dead_cross | (out["rsi"] > p["rsi_overbought"])

    out["buy_signal"] = out["buy_signal"].fillna(False)
    out["sell_signal"] = out["sell_signal"].fillna(False)

    return out, p
