"""
전략 신호를 받아 수수료를 반영한 가상 매매(전액 매수/매도, 단일 포지션)를 시뮬레이션한다.

lookahead(미래참조) 편향을 피하기 위해, N번째 봉에서 계산된 신호는
N+1번째 봉의 시가(open)에서 체결된다고 가정한다.
손절/익절은 보유 중 각 봉의 저가/고가가 기준가를 건드리면 그 기준가에 체결된 것으로 본다.
"""
import numpy as np
import pandas as pd

FEE_RATE = 0.001  # 바이낸스 테이커 수수료 근사치(0.1%)
INITIAL_CAPITAL = 10_000_000  # 원화 환산 기준 가상 시작 자본 (단위 무관, 비율만 의미 있음)
BARS_PER_YEAR = 24 * 365  # 1시간봉 기준


def run_backtest(signal_df: pd.DataFrame, params: dict, fee_rate: float = FEE_RATE,
                  initial_capital: float = INITIAL_CAPITAL) -> dict:
    df = signal_df.reset_index(drop=True)
    n = len(df)

    cash = initial_capital
    qty = 0.0
    position = None  # {"entry_price", "entry_time", "cost"}
    trades = []
    equity_points = np.empty(n)

    for i in range(n):
        row = df.iloc[i]

        if position is not None:
            stop_price = position["entry_price"] * (1 - params["stop_loss_pct"])
            target_price = position["entry_price"] * (1 + params["take_profit_pct"])
            exit_price, reason = None, None

            if row["low"] <= stop_price:
                exit_price, reason = stop_price, "stop_loss"
            elif row["high"] >= target_price:
                exit_price, reason = target_price, "take_profit"
            elif i > 0 and bool(df.iloc[i - 1]["sell_signal"]):
                exit_price, reason = row["open"], "signal_exit"

            if exit_price is not None:
                proceeds = qty * exit_price * (1 - fee_rate)
                pnl = proceeds - position["cost"]
                cash += proceeds
                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": row["open_time"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl": pnl,
                    "return_pct": (exit_price / position["entry_price"]) - 1,
                })
                position, qty = None, 0.0

        if position is None and i > 0 and bool(df.iloc[i - 1]["buy_signal"]):
            entry_price = row["open"]
            cost_basis = cash
            fee = cost_basis * fee_rate
            qty = (cost_basis - fee) / entry_price
            position = {"entry_price": entry_price, "entry_time": row["open_time"], "cost": cost_basis}
            cash = 0.0

        mark_price = row["close"]
        equity_points[i] = cash + (qty * mark_price if position is not None else 0.0)

    # 백테스트 종료 시점에 포지션이 열려 있으면 마지막 종가로 강제 청산
    if position is not None:
        last_row = df.iloc[-1]
        proceeds = qty * last_row["close"] * (1 - fee_rate)
        pnl = proceeds - position["cost"]
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": last_row["open_time"],
            "entry_price": position["entry_price"],
            "exit_price": last_row["close"],
            "reason": "end_of_period",
            "pnl": pnl,
            "return_pct": (last_row["close"] / position["entry_price"]) - 1,
        })
        cash = proceeds
        equity_points[-1] = cash

    equity = pd.Series(equity_points, index=df["open_time"])
    metrics = compute_metrics(equity, trades, initial_capital)
    return {"metrics": metrics, "trades": trades, "equity": equity}


def compute_metrics(equity: pd.Series, trades: list, initial_capital: float) -> dict:
    final_equity = equity.iloc[-1]
    total_return = (final_equity / initial_capital) - 1

    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

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

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe": round(float(sharpe), 2),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate * 100, 2),
        "final_equity": round(float(final_equity), 2),
    }
