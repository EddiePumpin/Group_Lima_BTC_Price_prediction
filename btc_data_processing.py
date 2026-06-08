from __future__ import annotations

from pathlib import Path

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
