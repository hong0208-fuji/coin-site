"""
전략 신호를 받아 수수료 + 펀딩비를 반영한 가상 매매(롱/숏 단일 포지션)를 시뮬레이션한다.

lookahead(미래참조) 편향을 피하기 위해, N번째 봉에서 계산된 신호는
N+1번째 봉의 시가(open)에서 체결된다고 가정한다.
손절/익절은 보유 중 각 봉의 저가/고가가 기준가를 건드리면 그 기준가에 체결된 것으로 본다.
숏 포지션은 1배(레버리지 없음) 기준 선형 근사로 손익을 계산하고, 실제 무기한 선물처럼
펀딩 정산 시점마다 펀딩비를 포지션 가치에서 가감한다(롱은 펀딩비가 양수면 지불, 숏은 반대).

포지션 사이징은 "매 거래 전액배팅"이 아니라 리스크 기반이다: 손절에 걸렸을 때
자본의 risk_per_trade_pct만큼만 잃도록 목표 포지션 크기(target_notional)를 역산한다
(target_notional = cash * risk_per_trade_pct / stop_loss_pct, 상한은 보유 현금 전액 = 레버리지 없음).
전액배팅은 승률이 애매한 전략을 많이 반복할 때 손실이 곱셈으로 누적되어 파산에
가까워지는 문제가 있어(자세한 배경은 docs/ARCHITECTURE.md v3 섹션 참고) 이렇게 바꿨다.

분할진입(선택 사항, initial_entry_fraction < 1.0일 때 활성화): 목표 포지션 크기 중
initial_entry_fraction만 먼저 진입하고, 이후 confirm_window봉 안에 진입봉의 고점(롱)
/저점(숏)을 돌파하는 확인이 나오면 나머지를 다음 봉 시가에 추가 진입한다(지지선 반등이든
추세 돌파든 결과적으로 "진입가 방향으로 새로운 극값을 만든다"는 공통점을 확인 조건으로 단순화).
확인이 안 나오면 초기 물량만으로 손절/익절을 관리한다. 손절/익절 기준가는 최초 진입가로 고정.
"""
import numpy as np
import pandas as pd

FEE_RATE = 0.001  # 바이낸스 테이커 수수료 근사치(0.1%)
INITIAL_CAPITAL = 10_000_000  # 가상 시작 자본 (단위 무관, 비율만 의미 있음)
BARS_PER_YEAR = 24 * 365  # 1시간봉 기준
DEFAULT_RISK_PER_TRADE_PCT = 0.01  # 손절 시 자본의 1%만 잃도록 포지션 크기 역산


def _open_tranche(cash, target_notional, entry_price, fee_rate):
    notional_gross = min(target_notional, cash)
    fee = notional_gross * fee_rate
    return {"entry_price": entry_price, "notional": notional_gross - fee}, notional_gross


def run_backtest(signal_df: pd.DataFrame, params: dict, fee_rate: float = FEE_RATE,
                  initial_capital: float = INITIAL_CAPITAL) -> dict:
    df = signal_df.reset_index(drop=True)
    n = len(df)
    risk_per_trade_pct = params.get("risk_per_trade_pct", DEFAULT_RISK_PER_TRADE_PCT)
    initial_entry_fraction = params.get("initial_entry_fraction", 1.0)
    confirm_window = params.get("confirm_window", 0)

    cash = initial_capital  # 포지션에 들어가지 않은 유휴 현금
    position = None
    trades = []
    equity_points = np.empty(n)

    for i in range(n):
        row = df.iloc[i]

        if position is not None:
            direction = position["direction"]
            total_notional = sum(t["notional"] for t in position["tranches"])

            # 펀딩 정산: 실제 정산 시점(is_funding_bar)에만 반영, 이중 적용 방지
            if bool(row.get("is_funding_bar", False)):
                funding_rate = float(row["funding_rate"])
                sign = 1 if direction == "LONG" else -1
                position["funding_accum"] += sign * funding_rate * total_notional

            # 확인 후 추가진입 처리 (분할진입 활성화 시)
            if position.get("pending_addon"):
                add_price = row["open"]
                remaining_target = position["addon_target_notional"]
                if remaining_target > 1e-9 and cash > 1e-9:
                    tranche, spent = _open_tranche(cash, remaining_target, add_price, fee_rate)
                    cash -= spent
                    position["tranches"].append(tranche)
                position["pending_addon"] = False
                position["confirmed"] = True

            if (
                not position["confirmed"]
                and not position.get("pending_addon")
                and confirm_window
                and (i - position["entry_index"]) <= confirm_window
            ):
                if direction == "LONG" and row["close"] > position["breakout_ref"]:
                    position["pending_addon"] = True
                elif direction == "SHORT" and row["close"] < position["breakout_ref"]:
                    position["pending_addon"] = True

            max_holding_bars = params.get("max_holding_bars")
            held_too_long = (
                max_holding_bars is not None and (i - position["entry_index"]) >= max_holding_bars
            )

            exit_price, reason = None, None
            if direction == "LONG":
                if row["low"] <= position["stop_price"]:
                    exit_price, reason = position["stop_price"], "stop_loss"
                elif row["high"] >= position["target_price"]:
                    exit_price, reason = position["target_price"], "take_profit"
                elif i > 0 and bool(df.iloc[i - 1]["sell_signal"]):
                    exit_price, reason = row["open"], "signal_exit"
                elif held_too_long:
                    exit_price, reason = row["open"], "max_holding"
            else:
                if row["high"] >= position["stop_price"]:
                    exit_price, reason = position["stop_price"], "stop_loss"
                elif row["low"] <= position["target_price"]:
                    exit_price, reason = position["target_price"], "take_profit"
                elif i > 0 and bool(df.iloc[i - 1]["cover_signal"]):
                    exit_price, reason = row["open"], "signal_exit"
                elif held_too_long:
                    exit_price, reason = row["open"], "max_holding"

            if exit_price is not None:
                direction_mult = 1 if direction == "LONG" else -1
                total_notional = sum(t["notional"] for t in position["tranches"])
                raw_pnl = sum(
                    t["notional"] * direction_mult * ((exit_price / t["entry_price"]) - 1)
                    for t in position["tranches"]
                )
                gross_value = total_notional + raw_pnl - position["funding_accum"]
                exit_fee = max(gross_value, 0) * fee_rate
                net_value = gross_value - exit_fee

                trades.append({
                    "direction": direction,
                    "entry_time": position["entry_time"],
                    "exit_time": row["open_time"],
                    "entry_price": position["tranches"][0]["entry_price"],
                    "exit_price": exit_price,
                    "reason": reason,
                    "num_tranches": len(position["tranches"]),
                    "funding_cost": position["funding_accum"],
                    "pnl": net_value - total_notional,
                    "return_pct": (net_value / total_notional) - 1 if total_notional else 0.0,
                })
                cash += net_value
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
                initial_target = target_notional * initial_entry_fraction
                tranche, spent = _open_tranche(cash, initial_target, entry_price, fee_rate)
                cash -= spent

                if entry_direction == "LONG":
                    stop_price = entry_price * (1 - params["stop_loss_pct"])
                    target_price = entry_price * (1 + params["take_profit_pct"])
                    breakout_ref = row["high"]
                else:
                    stop_price = entry_price * (1 + params["stop_loss_pct"])
                    target_price = entry_price * (1 - params["take_profit_pct"])
                    breakout_ref = row["low"]

                position = {
                    "direction": entry_direction, "entry_time": row["open_time"], "entry_index": i,
                    "tranches": [tranche], "funding_accum": 0.0,
                    "stop_price": stop_price, "target_price": target_price,
                    "breakout_ref": breakout_ref,
                    "addon_target_notional": max(target_notional - initial_target, 0.0),
                    "confirmed": initial_entry_fraction >= 1.0,
                    "pending_addon": False,
                }

        if position is not None:
            direction_mult = 1 if position["direction"] == "LONG" else -1
            total_notional = sum(t["notional"] for t in position["tranches"])
            unrealized = sum(
                t["notional"] * direction_mult * ((row["close"] / t["entry_price"]) - 1)
                for t in position["tranches"]
            )
            equity_points[i] = cash + total_notional + unrealized - position["funding_accum"]
        else:
            equity_points[i] = cash

    if position is not None:
        last_row = df.iloc[-1]
        direction = position["direction"]
        direction_mult = 1 if direction == "LONG" else -1
        total_notional = sum(t["notional"] for t in position["tranches"])
        raw_pnl = sum(
            t["notional"] * direction_mult * ((last_row["close"] / t["entry_price"]) - 1)
            for t in position["tranches"]
        )
        gross_value = total_notional + raw_pnl - position["funding_accum"]
        exit_fee = max(gross_value, 0) * fee_rate
        net_value = gross_value - exit_fee
        trades.append({
            "direction": direction,
            "entry_time": position["entry_time"],
            "exit_time": last_row["open_time"],
            "entry_price": position["tranches"][0]["entry_price"],
            "exit_price": last_row["close"],
            "reason": "end_of_period",
            "num_tranches": len(position["tranches"]),
            "funding_cost": position["funding_accum"],
            "pnl": net_value - total_notional,
            "return_pct": (net_value / total_notional) - 1 if total_notional else 0.0,
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
    addon_trades = [t for t in trades if t.get("num_tranches", 1) > 1]

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe": round(float(sharpe), 2),
        "num_trades": len(trades),
        "num_long_trades": len(long_trades),
        "num_short_trades": len(short_trades),
        "num_addon_trades": len(addon_trades),
        "win_rate_pct": round(win_rate * 100, 2),
        "total_funding_cost": round(float(total_funding_cost), 2),
        "final_equity": round(float(final_equity), 2),
    }
