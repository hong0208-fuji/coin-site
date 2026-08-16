"""
v6: "짧게짧게 수익실현" 스타일 검증.

v5의 분할진입(선진입 30% + 확인창 4봉)은 고정하고, 더 작은 익절폭(2~4%)과
짧은 최대보유시간(max_holding_bars)을 결합했을 때 더 나아지는지 정식 탐색한다.
risk_per_trade_pct는 1%로 고정(레버리지/공격적 베팅 효과는 optimize_v5의
리스크% 감도 테스트에서 이미 별도로 확인함 - 여기서는 진입/청산 스타일만 본다).
"""
import itertools
import json
from pathlib import Path

import pandas as pd

from strategy import generate_signals
from backtest import run_backtest, INITIAL_CAPITAL
from optimize_v5 import BEST_SIGNAL_PARAMS, split_train_test

DATA_FILE = Path(__file__).parent / "data" / "BTCUSDT_1h_3y.csv"
RESULTS_DIR = Path(__file__).parent / "results"

PARAM_GRID = {
    "initial_entry_fraction": [0.3],
    "confirm_window": [4],
    "stop_loss_pct": [0.02, 0.03],
    "take_profit_pct": [0.02, 0.03, 0.04, 0.08],
    "max_holding_bars": [12, 24, 48, None],
}


def param_combinations():
    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*PARAM_GRID.values()):
        yield dict(zip(keys, values))


def run_optimization():
    price_df = pd.read_csv(DATA_FILE, parse_dates=["open_time", "close_time"])
    combos = list(param_combinations())

    results = []
    cutoff = None
    for idx, entry_params in enumerate(combos, start=1):
        params = {**BEST_SIGNAL_PARAMS, **entry_params}
        signal_df, resolved = generate_signals(price_df, params)
        train_df, test_df, cutoff = split_train_test(signal_df)

        train_result = run_backtest(train_df, resolved)
        test_result = run_backtest(test_df, resolved)
        results.append({"params": resolved, "train": train_result["metrics"], "test": test_result["metrics"]})
        if idx % 10 == 0 or idx == len(combos):
            print(f"  {idx}/{len(combos)} 완료")

    results.sort(key=lambda r: (r["train"]["sharpe"], r["train"]["total_return_pct"]), reverse=True)
    return results, cutoff


def save_report(results, cutoff):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    top = results[:10]

    report = {
        "initial_capital": INITIAL_CAPITAL,
        "train_test_cutoff": str(cutoff.date()),
        "num_combinations_tested": len(results),
        "top_results": top,
    }
    (RESULTS_DIR / "report_v6_quick_scalp.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    lines = ["# 백테스트 결과 v6 - 짧게짧게 수익실현(퀵 스캘프) 스타일", ""]
    lines.append("v5의 분할진입(30% 선진입, 확인창 4봉)은 고정, 더 작은 익절폭 + 짧은 최대보유시간 조합 탐색.")
    lines.append(f"- train/test 분리 기준일: {cutoff.date()}")
    lines.append(f"- 탐색한 조합 수: {len(results)}")
    lines.append("")
    lines.append("## 상위 10개 (train 샤프비율 순)")
    lines.append("| 손절 | 익절 | 최대보유(봉) | train 수익률 | train 샤프 | test 수익률 | test 샤프 | 거래수 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in top:
        p, tr, te = r["params"], r["train"], r["test"]
        mh = p.get("max_holding_bars")
        mh_str = str(mh) if mh else "제한없음"
        lines.append(
            f"| {p['stop_loss_pct']} | {p['take_profit_pct']} | {mh_str} | {tr['total_return_pct']}% | "
            f"{tr['sharpe']} | {te['total_return_pct']}% | {te['sharpe']} | {te['num_trades']} |"
        )
    (RESULTS_DIR / "report_v6_quick_scalp.md").write_text("\n".join(lines))
    print(f"리포트 저장 완료: {RESULTS_DIR / 'report_v6_quick_scalp.json'}, {RESULTS_DIR / 'report_v6_quick_scalp.md'}")


if __name__ == "__main__":
    results, cutoff = run_optimization()
    save_report(results, cutoff)
