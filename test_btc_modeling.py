import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from btc_data_processing import (
    add_technical_features,
    add_time_features,
    build_feature_matrix,
    create_target,
    load_clean_coinmarketcap_csv,
    split_train_val_test,
)
from btc_modeling import run_pipeline


def test_feature_engineering_pipeline():
    csv_path = Path("btc_yfinance_2015_2026.csv")
    # Ensure the raw yfinance file contains expected columns
    raw = pd.read_csv(csv_path)
    assert any(c in raw.columns for c in ["Date", "Datetime", "date"]), "Expected Date/Datetime column in raw CSV."
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    assert required_cols.issubset(set(raw.columns)), f"Raw CSV missing required OHLCV columns: {required_cols - set(raw.columns)}"

    df = load_clean_coinmarketcap_csv(csv_path)
    df = add_time_features(df)
    df = add_technical_features(df)
    df = create_target(df, horizon=1)

    assert "prev_close" in df.columns
    assert "close_ma_5" in df.columns
    assert "rsi_14" in df.columns
    assert "target_1d" in df.columns
    assert df["target_1d"].notna().any(), "Expected some target values after shifting."

    X, y = build_feature_matrix(df, target_column="target_1d")
    assert len(X) > 0
    assert len(y) > 0
    assert X.shape[0] == y.shape[0]
    assert X["prev_close"].notna().all()

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert X_train.index.max() < X_val.index.min()
    assert X_val.index.max() < X_test.index.min()


def test_modeling_pipeline_runs():
    csv_path = Path("btc_yfinance_2015_2026.csv")
    results = run_pipeline(csv_path)
    assert "baseline" in results
    assert "linear_regression" in results
    assert "random_forest" in results
    assert results["baseline"]["mae"] >= 0
    assert results["linear_regression"]["rmse"] >= 0
    assert results["random_forest"]["r2"] <= 1.0


if __name__ == "__main__":
    test_feature_engineering_pipeline()
    test_modeling_pipeline_runs()
    print("Modeling pipeline tests passed.")
