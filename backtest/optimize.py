"""
전략 파라미터 조합을 그리드서치로 자동 탐색한다.

과최적화(overfitting)를 걸러내기 위해 3년 데이터를 앞 2년(train)과
최근 1년(test, out-of-sample)으로 나눈다. train 구간에서 샤프비율이
가장 좋은 파라미터 조합을 고른 뒤, 한 번도 학습에 쓰지 않은 test
구간에서 같은 파라미터로 재검증한다. train에서만 좋고 test에서
크게 나빠진다면 과최적화로 판단해야 한다.
"""
import itertools
import json
from pathlib import Path

import pandas as pd

from strategy import generate_signals
from backtest import run_backtest, INITIAL_CAPITAL

DATA_FILE = Path(__file__).parent / "data" / "BTCUSDT_1h_3y.csv"
RESULTS_DIR = Path(__file__).parent / "results"
TEST_PERIOD_DAYS = 365  # 최근 1년을 out-of-sample 검증 구간으로 사용

PARAM_GRID = {
    "short_window": [10, 20, 30],
    "long_window": [50, 100],
    "rsi_period": [14],
    "rsi_overbought": [65, 70, 75],
    "stop_loss_pct": [0.02, 0.03, 0.05],
    "take_profit_pct": [0.04, 0.06, 0.08],
}


def param_combinations():
    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*PARAM_GRID.values()):
        yield dict(zip(keys, values))


def split_train_test(df: pd.DataFrame):
    cutoff = df["open_time"].max() - pd.Timedelta(days=TEST_PERIOD_DAYS)
    train_df = df[df["open_time"] < cutoff].reset_index(drop=True)
    test_df = df[df["open_time"] >= cutoff].reset_index(drop=True)
    return train_df, test_df, cutoff


def run_optimization():
    df = pd.read_csv(DATA_FILE, parse_dates=["open_time", "close_time"])
    _, _, cutoff = split_train_test(df)
    combos = list(param_combinations())
    print(f"총 {len(combos)}개 파라미터 조합 탐색 (train < {cutoff.date()} / test >= {cutoff.date()})")

    results = []
    for idx, params in enumerate(combos, start=1):
        signal_df, resolved_params = generate_signals(df, params)
        train_df, test_df, _ = split_train_test(signal_df)

        if len(train_df) < 200 or len(test_df) < 50:
            continue

        train_result = run_backtest(train_df, resolved_params)
        test_result = run_backtest(test_df, resolved_params)

        results.append({
            "params": resolved_params,
            "train": train_result["metrics"],
            "test": test_result["metrics"],
        })

        if idx % 20 == 0 or idx == len(combos):
            print(f"  {idx}/{len(combos)} 완료")

    # train 샤프비율 기준 정렬, 동률이면 train 총수익률로 정렬
    results.sort(key=lambda r: (r["train"]["sharpe"], r["train"]["total_return_pct"]), reverse=True)
    return results, cutoff


def save_report(results: list, cutoff, top_n: int = 10):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    top = results[:top_n]

    report = {
        "initial_capital": INITIAL_CAPITAL,
        "train_test_cutoff": str(cutoff.date()),
        "num_combinations_tested": len(results),
        "best": top[0] if top else None,
        "top_results": top,
    }
    (RESULTS_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    lines = ["# 백테스트 결과 (3년치 BTCUSDT 1시간봉)", ""]
    lines.append(f"- 학습(train) / 검증(test) 분리 기준일: {cutoff.date()}")
    lines.append(f"- 탐색한 파라미터 조합 수: {len(results)}")
    lines.append("")
    lines.append("## 최적 파라미터 (train 샤프비율 기준 1위)")
    if top:
        best = top[0]
        lines.append(f"```\n{json.dumps(best['params'], ensure_ascii=False, indent=2)}\n```")
        lines.append("")
        lines.append("| 구간 | 총수익률 | CAGR | MDD | 샤프 | 거래수 | 승률 |")
        lines.append("|---|---|---|---|---|---|---|")
        for label, m in [("학습(train, ~2년)", best["train"]), ("검증(test, 최근 1년, out-of-sample)", best["test"])]:
            lines.append(
                f"| {label} | {m['total_return_pct']}% | {m['cagr_pct']}% | {m['max_drawdown_pct']}% | "
                f"{m['sharpe']} | {m['num_trades']} | {m['win_rate_pct']}% |"
            )
    lines.append("")
    lines.append("## 상위 10개 조합 (train 샤프비율 순)")
    lines.append("| 순위 | short/long/rsi_ob/sl/tp | train 수익률 | train 샤프 | test 수익률 | test 샤프 |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(top, start=1):
        p = r["params"]
        param_str = f"{p['short_window']}/{p['long_window']}/{p['rsi_overbought']}/{p['stop_loss_pct']}/{p['take_profit_pct']}"
        lines.append(
            f"| {i} | {param_str} | {r['train']['total_return_pct']}% | {r['train']['sharpe']} | "
            f"{r['test']['total_return_pct']}% | {r['test']['sharpe']} |"
        )
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))
    print(f"리포트 저장 완료: {RESULTS_DIR / 'report.json'}, {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    results, cutoff = run_optimization()
    save_report(results, cutoff)
