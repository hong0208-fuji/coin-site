"""
ML 기반 신호 생성 파이프라인.

3분할 검증으로 과최적화를 이중으로 막는다:
  1. model_train (~20개월): 그래디언트 부스팅 모델을 여기서만 학습한다.
  2. model_val (~4개월): 진입임계값/손절익절/보유기간 등 매매 파라미터를
     여기서 그리드서치로 자체탐색(피드백 루프)한다. 모델 재학습은 하지 않는다.
  3. test (최근 1년, out-of-sample): model_val에서 고른 단 하나의 조합을
     "한 번만" 검증한다. 이 구간 성과를 보고 파라미터를 다시 고치지 않는다
     (안 그러면 test조차 사실상 튜닝 대상이 되어버려 검증 의미가 없어짐).
"""
import itertools
import json
from pathlib import Path

import pandas as pd

from ml_features import build_features
from ml_model import train_model, predict
from ml_strategy import generate_ml_signals
from backtest import run_backtest, INITIAL_CAPITAL

DATA_FILE = Path(__file__).parent / "data" / "BTCUSDT_1h_3y.csv"
RESULTS_DIR = Path(__file__).parent / "results"

TEST_PERIOD_DAYS = 365
VAL_PERIOD_DAYS = 120

HORIZONS = [4, 8]

PARAM_GRID = {
    "entry_threshold_long": [0.0003, 0.0006, 0.001],
    "entry_threshold_short": [0.0003, 0.0006, 0.001],
    "stop_loss_pct": [0.01, 0.02],
    "take_profit_pct": [0.02, 0.03],
    "max_holding_bars": [8, 24],
}


def split_dates(df: pd.DataFrame):
    test_cutoff = df["open_time"].max() - pd.Timedelta(days=TEST_PERIOD_DAYS)
    val_cutoff = test_cutoff - pd.Timedelta(days=VAL_PERIOD_DAYS)
    return val_cutoff, test_cutoff


def param_combinations():
    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*PARAM_GRID.values()):
        yield dict(zip(keys, values))


def run_optimization():
    price_df = pd.read_csv(DATA_FILE, parse_dates=["open_time", "close_time"])
    feat_df = build_features(price_df)
    val_cutoff, test_cutoff = split_dates(feat_df)

    model_train_df = feat_df[feat_df["open_time"] < val_cutoff].reset_index(drop=True)
    print(f"model_train < {val_cutoff.date()} ({len(model_train_df)}행) / "
          f"model_val [{val_cutoff.date()}, {test_cutoff.date()}) / test >= {test_cutoff.date()}")

    val_results = []
    predictions_by_horizon = {}
    for horizon in HORIZONS:
        print(f"[horizon={horizon}] 모델 학습 중...")
        model = train_model(model_train_df, horizon=horizon)
        preds = predict(model, feat_df)
        predictions_by_horizon[horizon] = preds

        combos = list(param_combinations())
        for idx, params in enumerate(combos, start=1):
            params_full = {**params, "horizon": horizon}
            signal_df, resolved = generate_ml_signals(feat_df, preds, params_full)

            val_df = signal_df[(signal_df["open_time"] >= val_cutoff) & (signal_df["open_time"] < test_cutoff)].reset_index(drop=True)
            if len(val_df) < 100:
                continue
            val_result = run_backtest(val_df, resolved)
            val_results.append({"params": resolved, "val": val_result["metrics"]})

            if idx % 40 == 0 or idx == len(combos):
                print(f"  horizon={horizon}: {idx}/{len(combos)} 완료")

    val_results.sort(key=lambda r: (r["val"]["sharpe"], r["val"]["total_return_pct"]), reverse=True)
    best = val_results[0]
    best_horizon = best["params"]["horizon"]

    # 최종 검증: 고른 파라미터 단 하나로 held-out test 구간을 "한 번만" 평가
    final_signal_df, _ = generate_ml_signals(feat_df, predictions_by_horizon[best_horizon], best["params"])
    test_df = final_signal_df[final_signal_df["open_time"] >= test_cutoff].reset_index(drop=True)
    test_result = run_backtest(test_df, best["params"])

    # 참고용: train 구간 성과(모델이 직접 학습한 데이터라 참고치일 뿐, 선택 기준으로 쓰지 않음)
    train_df = final_signal_df[final_signal_df["open_time"] < val_cutoff].reset_index(drop=True)
    train_result = run_backtest(train_df, best["params"])

    return {
        "val_cutoff": val_cutoff,
        "test_cutoff": test_cutoff,
        "num_combinations_tested": len(val_results),
        "best_params": best["params"],
        "train_metrics": train_result["metrics"],
        "val_metrics": best["val"],
        "test_metrics": test_result["metrics"],
        "top_val_results": val_results[:10],
    }


def save_report(result: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "initial_capital": INITIAL_CAPITAL,
        "val_cutoff": str(result["val_cutoff"].date()),
        "test_cutoff": str(result["test_cutoff"].date()),
        "num_combinations_tested": result["num_combinations_tested"],
        "best_params": result["best_params"],
        "train_metrics": result["train_metrics"],
        "val_metrics": result["val_metrics"],
        "test_metrics": result["test_metrics"],
        "top_val_results": result["top_val_results"],
    }
    (RESULTS_DIR / "report_v3_ml.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    lines = ["# 백테스트 결과 v3 - ML(그래디언트 부스팅) 신호 생성 (3년치 BTCUSDT 1시간봉)", ""]
    lines.append("규칙 기반 크로스오버 대신, 다변수 피처(수익률/RSI/MACD/볼린저/변동성/거래량/펀딩비율/매크로 플래그/시간대)로")
    lines.append("향후 N봉 수익률을 예측하는 HistGradientBoostingRegressor를 사용. 3요인 필터(v2)는 진입 게이트로 계속 사용.")
    lines.append("")
    lines.append(f"- model_train: ~{result['val_cutoff'].date()} 이전 (모델 학습 전용, 파라미터 탐색에 사용 안 함)")
    lines.append(f"- model_val: {result['val_cutoff'].date()} ~ {result['test_cutoff'].date()} (진입임계값/손절익절/보유기간 자체탐색)")
    lines.append(f"- test: {result['test_cutoff'].date()} 이후 (한 번만 검증한 진짜 out-of-sample)")
    lines.append(f"- 탐색한 매매 파라미터 조합 수: {result['num_combinations_tested']} (horizon 2종 x 그리드)")
    lines.append("")
    lines.append("## 선택된 파라미터 (model_val 샤프비율 기준 1위)")
    lines.append(f"```\n{json.dumps(result['best_params'], ensure_ascii=False, indent=2)}\n```")
    lines.append("")
    lines.append("| 구간 | 총수익률 | CAGR | MDD | 샤프 | 거래수(롱/숏) | 승률 |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, m in [
        ("model_train (참고용, 학습에 쓰인 구간)", result["train_metrics"]),
        ("model_val (파라미터 선택에 쓰인 구간)", result["val_metrics"]),
        ("test (out-of-sample, 단 1회 검증)", result["test_metrics"]),
    ]:
        lines.append(
            f"| {label} | {m['total_return_pct']}% | {m['cagr_pct']}% | {m['max_drawdown_pct']}% | "
            f"{m['sharpe']} | {m['num_trades']}({m['num_long_trades']}/{m['num_short_trades']}) | {m['win_rate_pct']}% |"
        )
    (RESULTS_DIR / "report_v3_ml.md").write_text("\n".join(lines))
    print(f"리포트 저장 완료: {RESULTS_DIR / 'report_v3_ml.json'}, {RESULTS_DIR / 'report_v3_ml.md'}")


if __name__ == "__main__":
    result = run_optimization()
    save_report(result)
