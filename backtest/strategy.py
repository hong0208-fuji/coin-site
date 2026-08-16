"""
파라미터화된 이동평균 크로스 + RSI 필터 전략 (롱/숏 양방향).

롱 진입: 골든크로스 + RSI가 과매수 기준 이하 + (펀딩비율 롱 과열 아님/FOMC 전후 아님/변동성 급등 아님)
롱 청산: 데드크로스 또는 RSI 과매수 기준 초과
숏 진입: 데드크로스 + RSI가 과매도 기준 이상 + (펀딩비율 숏 과열 아님/FOMC 전후 아님/변동성 급등 아님)
숏 청산: 골든크로스 또는 RSI 과매도 기준 미만

손절/익절은 backtest.py에서 진입가 대비 등락률로 별도 체크한다.
시장 컨텍스트 필터(펀딩비율/매크로/변동성)는 진입에만 적용하고 청산은 지표 신호를 그대로 따른다
(포지션을 보호/청산하는 신호를 필터로 막으면 오히려 리스크가 커지기 때문).
"""
from indicators import add_indicators
from market_context import add_market_context

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
    out = add_market_context(out)

    prev_short = out["sma_short"].shift(1)
    prev_long = out["sma_long"].shift(1)

    golden_cross = (prev_short <= prev_long) & (out["sma_short"] > out["sma_long"])
    dead_cross = (prev_short >= prev_long) & (out["sma_short"] < out["sma_long"])

    out["buy_signal"] = (
        golden_cross & (out["rsi"] < p["rsi_overbought"]) & out["allow_long_entry"]
    ).fillna(False)
    out["sell_signal"] = (dead_cross | (out["rsi"] > p["rsi_overbought"])).fillna(False)

    out["short_signal"] = (
        dead_cross & (out["rsi"] > p["rsi_oversold"]) & out["allow_short_entry"]
    ).fillna(False)
    out["cover_signal"] = (golden_cross | (out["rsi"] < p["rsi_oversold"])).fillna(False)

    return out, p
