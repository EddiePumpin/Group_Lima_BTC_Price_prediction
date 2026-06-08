# Bitcoin Price Prediction Data Processing

This branch adds a dedicated data processing module for the CoinMarketCap-style Bitcoin dataset and removes the old unused dataset.

## What was added

- `btc_data_processing.py`
  - `load_clean_coinmarketcap_csv(csv_path)` loads the file `Bitcoin_5_11_2020-5_11_2026_historical_data_coinmarketcap.csv`
  - converts timestamps to pandas datetimes
  - converts price, volume, market cap, and supply fields to numeric values
  - removes duplicate timestamps
  - drops rows missing required values
  - adds a `date` column for easier daily analysis

- `test_btc_data_processing.py`
  - validates that the loader works
  - checks the cleaned dataset has the expected columns and data types

## What was removed

- `bitcoin-historical-data.csv`
  - this file is no longer used by the capstone project

## How to run the test

```powershell
cd "c:\Users\hp\Desktop\Group Lima\Group_Lima_BTC_Price_prediction"
C:/Users/hp/AppData/Local/Programs/Python/Python313/python.exe test_btc_data_processing.py
```

## Notes

- The branch is focused on the CoinMarketCap file only
- `btc_data_processing.py` is now the main data loader/cleaner for the project
