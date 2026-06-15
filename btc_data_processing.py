from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_clean_coinmarketcap_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load and clean the CoinMarketCap-style Bitcoin CSV file.

    This function is designed for the file named like
    `Bitcoin_5_11_2020-5_11_2026_historical_data_coinmarketcap.csv`.
    It parses datetimes, converts numeric fields, normalizes column names,
    removes duplicates, and drops rows with invalid core values.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        quotechar='"',
        parse_dates=["timestamp", "timeOpen", "timeClose", "timeHigh", "timeLow"],
        low_memory=False,
    )

    df = df.rename(columns={
        "timeOpen": "time_open",
        "timeClose": "time_close",
        "timeHigh": "time_high",
        "timeLow": "time_low",
        "name": "asset_name",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "marketCap": "market_cap",
        "circulatingSupply": "circulating_supply",
        "timestamp": "timestamp",
    })

    numeric_cols = ["open", "high", "low", "close", "volume", "market_cap", "circulating_supply"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "asset_name" in df.columns and df["asset_name"].nunique() == 1:
        df = df.drop(columns=["asset_name"])

    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep only rows with valid timestamp and close price
    required_cols = ["timestamp", "close"]
    df = df.dropna(subset=required_cols)

    df["date"] = df["timestamp"].dt.date

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-derived features to the cleaned dataset."""
    df = df.copy()
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and trend features for modeling."""
    df = df.copy()
    windows = [5, 10, 20]
    for window in windows:
        df[f"close_ma_{window}"] = df["close"].rolling(window).mean()
        df[f"close_std_{window}"] = df["close"].rolling(window).std()
        df[f"volume_ma_{window}"] = df["volume"].rolling(window).mean()

    df["close_pct_change"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["prev_close"] = df["close"].shift(1)
    df["prev_volume"] = df["volume"].shift(1)
    df["high_low_spread"] = df["high"] - df["low"]
    if "open" in df.columns:
        df["open_close_spread"] = df["close"] - df["open"]

    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    average_gain = pd.Series(gain).rolling(14).mean()
    average_loss = pd.Series(loss).rolling(14).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    return df


def create_target(df: pd.DataFrame, horizon: int = 1, target_column: str = "close") -> pd.DataFrame:
    """Create the prediction target for a future horizon."""
    df = df.copy()
    target_name = f"target_{horizon}d"
    target_return_name = f"target_return_{horizon}d"
    df[target_name] = df[target_column].shift(-horizon)
    df[target_return_name] = df[target_name] / df[target_column] - 1
    return df


def build_feature_matrix(df: pd.DataFrame, feature_columns: list[str] | None = None, target_column: str = "target_1d") -> tuple[pd.DataFrame, pd.Series]:
    """Select features and target values for modeling."""
    if feature_columns is None:
        feature_columns = [
            "prev_close",
            "prev_volume",
            "close_ma_5",
            "close_ma_10",
            "close_ma_20",
            "close_std_20",
            "volume_ma_5",
            "close_pct_change",
            "log_return",
            "high_low_spread",
            "open_close_spread",
            "rsi_14",
            "day_of_week",
            "month",
            "is_weekend",
        ]

    missing_columns = set(feature_columns + [target_column]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns for feature matrix: {sorted(missing_columns)}")

    X = df[feature_columns].copy()
    y = df[target_column].copy()
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split features and targets into time-series train/validation/test sets."""
    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size and test_size must be between 0 and 1.")

    n = len(X)
    test_n = max(1, int(n * test_size))
    val_n = max(1, int(n * val_size))
    train_end = n - test_n - val_n
    if train_end < 1:
        raise ValueError("Not enough observations for the requested split sizes.")

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end : n - test_n]
    X_test = X.iloc[n - test_n :]
    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end : n - test_n]
    y_test = y.iloc[n - test_n :]
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_dataset(csv_path: str | Path, horizon: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load raw data and return a prepared dataset ready for modeling."""
    df = load_clean_coinmarketcap_csv(csv_path)
    df = add_time_features(df)
    df = add_technical_features(df)
    df = create_target(df, horizon=horizon)
    X, y = build_feature_matrix(df, target_column=f"target_{horizon}d")
    return df, X, y


def validate_cleaned_dataframe(df: pd.DataFrame) -> None:
    """Validate the cleaned dataset for expected structure and core values."""
    expected_columns = {"timestamp", "open", "high", "low", "close", "volume", "market_cap", "circulating_supply", "date"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns after cleaning: {sorted(missing)}")

    if df["timestamp"].isna().any():
        raise ValueError("Cleaned data contains missing timestamps.")
    if df["close"].isna().any():
        raise ValueError("Cleaned data contains missing close prices.")

    if len(df) == 0:
        raise ValueError("Cleaned DataFrame is empty.")


def summary_report(df: pd.DataFrame) -> str:
    """Return a short summary of the cleaned DataFrame."""
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    non_null = df["close"].notna().sum()
    return (
        f"Rows: {len(df)}\n"
        f"Date range: {start_date.date()} to {end_date.date()}\n"
        f"Non-null close prices: {non_null}\n"
        f"Columns: {sorted(df.columns)}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python btc_data_processing.py <path-to-csv>")

    path_str = sys.argv[1]
    cleaned = load_clean_coinmarketcap_csv(path_str)
    validate_cleaned_dataframe(cleaned)
    print(cleaned.head())
    print(summary_report(cleaned))
