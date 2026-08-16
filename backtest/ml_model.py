"""
그래디언트 부스팅 회귀 모델로 향후 N봉 수익률을 예측한다.
결측 피처(지표 워밍업 구간 등)를 자동으로 처리하는 HistGradientBoostingRegressor 사용.
"""
from sklearn.ensemble import HistGradientBoostingRegressor

from ml_features import FEATURE_COLUMNS

MODEL_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "max_iter": 200,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "random_state": 42,
}


def train_model(train_df, horizon: int):
    from ml_features import build_labels

    labels = build_labels(train_df, horizon)
    valid = labels.notna() & train_df[FEATURE_COLUMNS].notna().all(axis=1)

    X = train_df.loc[valid, FEATURE_COLUMNS]
    y = labels.loc[valid]

    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    return model


def predict(model, df):
    X = df[FEATURE_COLUMNS]
    preds = model.predict(X)
    return preds
