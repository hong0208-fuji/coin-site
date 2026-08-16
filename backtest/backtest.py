"""
전략 신호를 받아 수수료 + 펀딩비를 반영한 가상 매매(롱/숏 단일 포지션)를 시뮬레이션한다.

lookahead(미래참조) 편향을 피하기 위해, N번째 봉에서 계산된 신호는
N+1번째 봉의 시가(open)에서 체결된다고 가정한다.
손절/익절은 보유 중 각 봉의 저가/고가가 기준가를 건드리면 그 기준가에 체결된 것으로 본다.
숏 포지션은 1배(레버리지 없음) 기준 선형 근사로 손익을 계산하고, 실제 무기한 선물처럼
펀딩 정산 시점마다 펀딩비를 포지션 가치에서 가감한다(롱은 펀딩비가 양수면 지불, 숏은 반대).
"""
import numpy as np
import pandas as pd

FEE_RATE = 0.001  # 바이낸스 테이커 수수료 근사치(0.1%)
INITIAL_CAPITAL = 10_000_000  # 가상 시작 자본 (단위 무관, 비율만 의미 있음)
BARS_PER_YEAR = 24 * 365  # 1시간봉 기준


def run_backtest(signal_df: pd.DataFrame, params: dict, fee_rate: float = FEE_RATE,
                  initial_capital: float = INITIAL_CAPITAL) -> dict:
    df = signal_df.reset_index(drop=True)
    n = len(df)

    cash = initial_capital
    position = None  # {"direction": "LONG"/"SHORT", "entry_price", "entry_time", "notional", "funding_accum"}
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
                net_cash = gross_value - exit_fee

                trades.append({
                    "direction": direction,
                    "entry_time": position["entry_time"],
                    "exit_time": row["open_time"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": reason,
                    "funding_cost": position["funding_accum"],
                    "pnl": net_cash - position["notional"],
                    "return_pct": (net_cash / position["notional"]) - 1,
                })
                cash = net_cash
                position = None

        if position is None and i > 0:
            prev = df.iloc[i - 1]
            if bool(prev["buy_signal"]):
                entry_price = row["open"]
                fee = cash * fee_rate
                position = {
                    "direction": "LONG", "entry_price": entry_price, "entry_time": row["open_time"],
                    "notional": cash - fee, "funding_accum": 0.0, "entry_index": i,
                }
                cash = 0.0
            elif bool(prev["short_signal"]):
                entry_price = row["open"]
                fee = cash * fee_rate
                position = {
                    "direction": "SHORT", "entry_price": entry_price, "entry_time": row["open_time"],
                    "notional": cash - fee, "funding_accum": 0.0, "entry_index": i,
                }
                cash = 0.0

        if position is not None:
            direction_mult = 1 if position["direction"] == "LONG" else -1
            unrealized = position["notional"] * direction_mult * ((row["close"] / position["entry_price"]) - 1)
            equity_points[i] = position["notional"] + unrealized - position["funding_accum"]
        else:
            equity_points[i] = cash

    if position is not None:
        last_row = df.iloc[-1]
        direction = position["direction"]
        direction_mult = 1 if direction == "LONG" else -1
        raw_pnl = position["notional"] * direction_mult * ((last_row["close"] / position["entry_price"]) - 1)
        gross_value = position["notional"] + raw_pnl - position["funding_accum"]
        exit_fee = max(gross_value, 0) * fee_rate
        net_cash = gross_value - exit_fee
        trades.append({
            "direction": direction,
            "entry_time": position["entry_time"],
            "exit_time": last_row["open_time"],
            "entry_price": position["entry_price"],
            "exit_price": last_row["close"],
            "reason": "end_of_period",
            "funding_cost": position["funding_accum"],
            "pnl": net_cash - position["notional"],
            "return_pct": (net_cash / position["notional"]) - 1,
        })
        cash = net_cash
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
