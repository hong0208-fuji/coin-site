"""
전략 신호를 받아 수수료 + 펀딩비를 반영한 가상 매매(롱/숏 단일 포지션)를 시뮬레이션한다.

lookahead(미래참조) 편향을 피하기 위해, N번째 봉에서 계산된 신호는
N+1번째 봉의 시가(open)에서 체결된다고 가정한다.
손절/익절은 보유 중 각 봉의 저가/고가가 기준가를 건드리면 그 기준가에 체결된 것으로 본다.
숏 포지션은 1배(레버리지 없음) 기준 선형 근사로 손익을 계산하고, 실제 무기한 선물처럼
펀딩 정산 시점마다 펀딩비를 포지션 가치에서 가감한다(롱은 펀딩비가 양수면 지불, 숏은 반대).

포지션 사이징은 "매 거래 전액배팅"이 아니라 리스크 기반이다: 손절에 걸렸을 때
자본의 risk_per_trade_pct만큼만 잃도록 포지션 크기를 역산한다
(notional = cash * risk_per_trade_pct / stop_loss_pct, 상한은 보유 현금 전액 = 레버리지 없음).
전액배팅은 승률이 애매한 전략을 많이 반복할 때 손실이 곱셈으로 누적되어 파산에
가까워지는 문제가 있어(자세한 배경은 docs/ARCHITECTURE.md v3 섹션 참고) 이렇게 바꿨다.
"""
import numpy as np
import pandas as pd

FEE_RATE = 0.001  # 바이낸스 테이커 수수료 근사치(0.1%)
INITIAL_CAPITAL = 10_000_000  # 가상 시작 자본 (단위 무관, 비율만 의미 있음)
BARS_PER_YEAR = 24 * 365  # 1시간봉 기준
DEFAULT_RISK_PER_TRADE_PCT = 0.01  # 손절 시 자본의 1%만 잃도록 포지션 크기 역산


def run_backtest(signal_df: pd.DataFrame, params: dict, fee_rate: float = FEE_RATE,
                  initial_capital: float = INITIAL_CAPITAL) -> dict:
    df = signal_df.reset_index(drop=True)
    n = len(df)
    risk_per_trade_pct = params.get("risk_per_trade_pct", DEFAULT_RISK_PER_TRADE_PCT)

    cash = initial_capital  # 포지션에 들어가지 않은 유휴 현금
    position = None  # {"direction", "entry_price", "entry_time", "entry_index", "notional", "funding_accum"}
    trades = []
    equity_points = np.empty(n)

    for i in range(n):
        row = df.iloc[i]

        if position is not None:
            direction = position["direction"]
            entry_price = position["entry_price"]

            if direction == "LONG":
                stop_price = entry_price * (1 - params["stop_loss_pct"])
                target_price = entry_price * (1 + params["take_profit_pct"])
            else:  # SHORT
                stop_price = entry_price * (1 + params["stop_loss_pct"])
                target_price = entry_price * (1 - params["take_profit_pct"])

            # 펀딩 정산: 실제 정산 시점(is_funding_bar)에만 반영, 이중 적용 방지
            if bool(row.get("is_funding_bar", False)):
                funding_rate = float(row["funding_rate"])
                sign = 1 if direction == "LONG" else -1  # 롱은 양수 펀딩비를 지불(비용), 숏은 반대
                position["funding_accum"] += sign * funding_rate * position["notional"]

            max_holding_bars = params.get("max_holding_bars")
            held_too_long = (
                max_holding_bars is not None and (i - position["entry_index"]) >= max_holding_bars
            )

            exit_price, reason = None, None
            if direction == "LONG":
                if row["low"] <= stop_price:
                    exit_price, reason = stop_price, "stop_loss"
                elif row["high"] >= target_price:
                    exit_price, reason = target_price, "take_profit"
                elif i > 0 and bool(df.iloc[i - 1]["sell_signal"]):
                    exit_price, reason = row["open"], "signal_exit"
                elif held_too_long:
                    exit_price, reason = row["open"], "max_holding"
            else:
                if row["high"] >= stop_price:
                    exit_price, reason = stop_price, "stop_loss"
                elif row["low"] <= target_price:
                    exit_price, reason = target_price, "take_profit"
                elif i > 0 and bool(df.iloc[i - 1]["cover_signal"]):
                    exit_price, reason = row["open"], "signal_exit"
                elif held_too_long:
                    exit_price, reason = row["open"], "max_holding"

            if exit_price is not None:
                direction_mult = 1 if direction == "LONG" else -1
                raw_pnl = position["notional"] * direction_mult * ((exit_price / entry_price) - 1)
                gross_value = position["notional"] + raw_pnl - position["funding_accum"]
                exit_fee = max(gross_value, 0) * fee_rate
                net_value = gross_value - exit_fee

                trades.append({
                    "direction": direction,
                    "entry_time": position["entry_time"],
                    "exit_time": row["open_time"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": reason,
                    "funding_cost": position["funding_accum"],
                    "pnl": net_value - position["notional"],
                    "return_pct": (net_value / position["notional"]) - 1,
                })
                cash += net_value  # 포지션에 넣었던 몫을 손익 반영해서 유휴 현금 풀로 반환
                position = None

        if position is None and i > 0:
            prev = df.iloc[i - 1]
            entry_direction = None
            if bool(prev["buy_signal"]):
                entry_direction = "LONG"
            elif bool(prev["short_signal"]):
                entry_direction = "SHORT"

            if entry_direction is not None:
                entry_price = row["open"]
                target_notional = cash * (risk_per_trade_pct / params["stop_loss_pct"])
                notional_gross = min(target_notional, cash)  # 레버리지 없음: 보유 현금 이상 투입 불가
                fee = notional_gross * fee_rate
                cash -= notional_gross  # 투입분만 유휴 현금 풀에서 차감(나머지는 유휴 현금으로 남음)
                position = {
                    "direction": entry_direction, "entry_price": entry_price, "entry_time": row["open_time"],
                    "notional": notional_gross - fee, "funding_accum": 0.0, "entry_index": i,
                }

        if position is not None:
            direction_mult = 1 if position["direction"] == "LONG" else -1
            unrealized = position["notional"] * direction_mult * ((row["close"] / position["entry_price"]) - 1)
            equity_points[i] = cash + position["notional"] + unrealized - position["funding_accum"]
        else:
            equity_points[i] = cash

    if position is not None:
        last_row = df.iloc[-1]
        direction = position["direction"]
        direction_mult = 1 if direction == "LONG" else -1
        raw_pnl = position["notional"] * direction_mult * ((last_row["close"] / position["entry_price"]) - 1)
        gross_value = position["notional"] + raw_pnl - position["funding_accum"]
        exit_fee = max(gross_value, 0) * fee_rate
        net_value = gross_value - exit_fee
        trades.append({
            "direction": direction,
            "entry_time": position["entry_time"],
            "exit_time": last_row["open_time"],
            "entry_price": position["entry_price"],
            "exit_price": last_row["close"],
            "reason": "end_of_period",
            "funding_cost": position["funding_accum"],
            "pnl": net_value - position["notional"],
            "return_pct": (net_value / position["notional"]) - 1,
        })
        cash += net_value
        equity_points[-1] = cash

    equity = pd.Series(equity_points, index=df["open_time"])
    metrics = compute_metrics(equity, trades, initial_capital)
    return {"metrics": metrics, "trades": trades, "equity": equity}


def compute_metrics(equity: pd.Series, trades: list, initial_capital: float) -> dict:
    final_equity = equity.iloc[-1]
    total_return = (final_equity / initial_capital) - 1

    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 and final_equity > 0 else -1.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    bar_returns = equity.pct_change().dropna()
    if bar_returns.std() > 0:
        sharpe = (bar_returns.mean() / bar_returns.std()) * np.sqrt(BARS_PER_YEAR)
    else:
        sharpe = 0.0

    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = (len(wins) / len(trades)) if trades else 0.0
    long_trades = [t for t in trades if t["direction"] == "LONG"]
    short_trades = [t for t in trades if t["direction"] == "SHORT"]
    total_funding_cost = sum(t["funding_cost"] for t in trades)

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe": round(float(sharpe), 2),
        "num_trades": len(trades),
        "num_long_trades": len(long_trades),
        "num_short_trades": len(short_trades),
        "win_rate_pct": round(win_rate * 100, 2),
        "total_funding_cost": round(float(total_funding_cost), 2),
        "final_equity": round(float(final_equity), 2),
    }
