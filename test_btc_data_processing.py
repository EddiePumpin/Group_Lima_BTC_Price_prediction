import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from btc_data_processing import load_clean_coinmarketcap_csv, validate_cleaned_dataframe


def test_coinmarketcap_csv_loads_and_cleans():
    csv_path = Path("Bitcoin_5_11_2020-5_11_2026_historical_data_coinmarketcap.csv")
    df = load_clean_coinmarketcap_csv(csv_path)
    validate_cleaned_dataframe(df)

    assert len(df) > 0, "Expected at least one row after cleaning."
    assert "timestamp" in df.columns
    assert "close" in df.columns
    assert df["close"].dtype.kind in "fiu", "Close price should be numeric."
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]) or pd.api.types.is_datetime64tz_dtype(df["timestamp"]), \
        "Timestamp column should be datetime."


if __name__ == "__main__":
    test_coinmarketcap_csv_loads_and_cleans()
    print("Test passed: CoinMarketCap CSV loads and cleans correctly.")
