"""
v5: v2의 최적 신호(규칙기반, short=30/long=100/rsi_ob=75/rsi_os=30)를 그대로 쓰고,
분할진입(초기 20~30% 선진입 -> 확인 후 나머지 진입)만 추가해서 효과를 검증한다.
신호 생성 로직은 이미 train 구간에서 골라낸 값으로 고정해 새로운 과최적화를 만들지 않는다.
"""
import itertools
import json
from pathlib import Path

import pandas as pd

from strategy import generate_signals
from backtest import run_backtest, INITIAL_CAPITAL

DATA_FILE = Path(__file__).parent / "data" / "BTCUSDT_1h_3y.csv"
RESULTS_DIR = Path(__file__).parent / "results"
TEST_PERIOD_DAYS = 365

BEST_SIGNAL_PARAMS = {
    "short_window": 30,
    "long_window": 100,
    "rsi_period": 14,
    "rsi_overbought": 75,
    "rsi_oversold": 30,
}

PARAM_GRID = {
    "initial_entry_fraction": [0.2, 0.3, 1.0],  # 1.0 = 분할진입 없음(비교 기준선)
    "confirm_window": [4, 8],
    "stop_loss_pct": [0.02, 0.03],
    "take_profit_pct": [0.04, 0.08],
}


def param_combinations():
    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*PARAM_GRID.values()):
        combo = dict(zip(keys, values))
        if combo["initial_entry_fraction"] >= 1.0:
            combo["confirm_window"] = 0  # 분할진입 없으면 확인창 의미 없음(중복 제거)
        yield combo


def split_train_test(df: pd.DataFrame):
    cutoff = df["open_time"].max() - pd.Timedelta(days=TEST_PERIOD_DAYS)
    return df[df["open_time"] < cutoff].reset_index(drop=True), df[df["open_time"] >= cutoff].reset_index(drop=True), cutoff


def run_optimization():
    price_df = pd.read_csv(DATA_FILE, parse_dates=["open_time", "close_time"])

    seen = set()
    combos = []
    for c in param_combinations():
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            combos.append(c)

    results = []
    cutoff = None
    for idx, entry_params in enumerate(combos, start=1):
        params = {**BEST_SIGNAL_PARAMS, **entry_params}
        signal_df, resolved = generate_signals(price_df, params)
        train_df, test_df, cutoff = split_train_test(signal_df)

        train_result = run_backtest(train_df, resolved)
        test_result = run_backtest(test_df, resolved)
        results.append({"params": resolved, "train": train_result["metrics"], "test": test_result["metrics"]})
        print(f"  {idx}/{len(combos)}: frac={entry_params['initial_entry_fraction']} "
              f"confirm={entry_params['confirm_window']} sl={entry_params['stop_loss_pct']} "
              f"tp={entry_params['take_profit_pct']} -> train={train_result['metrics']['total_return_pct']}% "
              f"test={test_result['metrics']['total_return_pct']}%")

    results.sort(key=lambda r: (r["train"]["sharpe"], r["train"]["total_return_pct"]), reverse=True)
    return results, cutoff


def save_report(results, cutoff):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline = next(r for r in results if r["params"]["initial_entry_fraction"] >= 1.0
                     and r["params"]["stop_loss_pct"] == 0.02 and r["params"]["take_profit_pct"] == 0.08)
    best = results[0]

    report = {
        "initial_capital": INITIAL_CAPITAL,
        "train_test_cutoff": str(cutoff.date()),
        "signal_params": BEST_SIGNAL_PARAMS,
        "num_combinations_tested": len(results),
        "baseline_no_scaled_entry": baseline,
        "best": best,
        "all_results": results,
    }
    (RESULTS_DIR / "report_v5_scaled_entry.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    lines = ["# 백테스트 결과 v5 - 분할진입(선진입 20~30% + 확인 후 추가진입)", ""]
    lines.append("v2의 최적 신호(short=30/long=100/rsi_ob=75/rsi_os=30)를 고정하고, 분할진입 여부/파라미터만 비교.")
    lines.append(f"- train/test 분리 기준일: {cutoff.date()}")
    lines.append(f"- 탐색한 조합 수: {len(results)}")
    lines.append("")
    lines.append("## 분할진입 없음(기준선) vs 있음(train 샤프비율 1위)")
    lines.append("| | 초기비율 | 확인창 | 손절 | 익절 | train 수익률 | train 샤프 | test 수익률 | test 샤프 | 추가진입된 거래수 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for label, r in [("분할진입 없음", baseline), ("분할진입 최적", best)]:
        p, tr, te = r["params"], r["train"], r["test"]
        lines.append(
            f"| {label} | {p['initial_entry_fraction']} | {p['confirm_window']} | {p['stop_loss_pct']} | "
            f"{p['take_profit_pct']} | {tr['total_return_pct']}% | {tr['sharpe']} | {te['total_return_pct']}% | "
            f"{te['sharpe']} | {te['num_addon_trades']}/{te['num_trades']} |"
        )
    lines.append("")
    lines.append("## 전체 조합 (train 샤프비율 순)")
    lines.append("| 초기비율 | 확인창 | 손절 | 익절 | train 수익률 | train 샤프 | test 수익률 | test 샤프 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        p, tr, te = r["params"], r["train"], r["test"]
        lines.append(
            f"| {p['initial_entry_fraction']} | {p['confirm_window']} | {p['stop_loss_pct']} | "
            f"{p['take_profit_pct']} | {tr['total_return_pct']}% | {tr['sharpe']} | {te['total_return_pct']}% | {te['sharpe']} |"
        )
    (RESULTS_DIR / "report_v5_scaled_entry.md").write_text("\n".join(lines))
    print(f"리포트 저장 완료: {RESULTS_DIR / 'report_v5_scaled_entry.json'}, {RESULTS_DIR / 'report_v5_scaled_entry.md'}")


if __name__ == "__main__":
    results, cutoff = run_optimization()
    save_report(results, cutoff)
