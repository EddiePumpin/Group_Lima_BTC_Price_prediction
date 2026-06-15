from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

from btc_data_processing import (
    build_feature_matrix,
    create_target,
    load_clean_coinmarketcap_csv,
    split_train_val_test,
    add_technical_features,
    add_time_features,
)


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def train_linear_regression(X: np.ndarray, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_random_forest(X: np.ndarray, y: pd.Series, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X, y)
    return model


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mse),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def baseline_previous_close(X_test: pd.DataFrame) -> np.ndarray:
    if "prev_close" not in X_test.columns:
        raise ValueError("Baseline requires feature column 'prev_close'.")
    return X_test["prev_close"].to_numpy()


def run_pipeline(
    csv_path: str | Path,
    horizon: int = 1,
    val_size: float = 0.05,
    test_size: float = 0.05,
) -> dict[str, dict[str, float]]:
    raw_df = load_clean_coinmarketcap_csv(csv_path)
    df = add_time_features(raw_df)
    df = add_technical_features(df)
    df = create_target(df, horizon=horizon)
    X, y = build_feature_matrix(df, target_column=f"target_{horizon}d")

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
    X, y, val_size=val_size, test_size=test_size
)
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train, X_val, X_test
    )

    baseline_pred = baseline_previous_close(X_test)
    baseline_metrics = evaluate_regression(y_test, baseline_pred)

    linear_model = train_linear_regression(X_train_scaled, y_train)
    linear_pred = linear_model.predict(X_test_scaled)
    linear_metrics = evaluate_regression(y_test, linear_pred)

    rf_model = train_random_forest(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluate_regression(y_test, rf_pred)

    results = {
        "baseline": baseline_metrics,
        "linear_regression": linear_metrics,
        "random_forest": rf_metrics,
    }

    return results


def print_results(results: dict[str, dict[str, float]]) -> None:
    print("Model evaluation results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python btc_modeling.py <path-to-csv>"
        )

    csv_path = sys.argv[1]
    results = run_pipeline(csv_path)
    print_results(results)
