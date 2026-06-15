# Bitcoin Price Prediction Data Processing

This project implements a Bitcoin daily close price prediction workflow using yfinance historical data (2015–2026, 4000+ entries).
The pipeline includes CSV loading and cleaning, feature engineering, target creation, time-series splits, model training, and evaluation.

## Project files

- `fetch_btc_data.py`
  - Downloads Bitcoin data from yfinance and exports to CSV (semicolon-delimited)
  - Used to generate `btc_yfinance_2015_2026.csv`

- `btc_data_processing.py`
  - loads and cleans `btc_yfinance_2015_2026.csv` (yfinance export, semicolon-delimited)
  - parses timestamps and numeric values
  - removes duplicate timestamps and invalid rows
  - adds time features and technical indicators
  - creates a one-day-ahead close price target
  - builds a feature matrix and time-series train/validation/test split

- `btc_modeling.py`
  - scales the feature set with `StandardScaler`
  - trains a baseline previous-close predictor, linear regression, and random forest
  - evaluates models using MAE, RMSE, MAPE, and R²

- `test_btc_data_processing.py`
  - validates CSV loading, data cleaning, and expected timestamp/close columns

- `test_btc_modeling.py`
  - validates feature engineering, splitting, and modeling pipeline execution

- `requirements.txt`
  - Python dependencies: `pandas`, `numpy`, `scikit-learn`, `yfinance`

- `btc_yfinance_analysis.ipynb`
  - Jupyter notebook with runnable cells for data exploration and modeling
  - loads raw yfinance CSV, runs the full pipeline, and displays model results

## Current pipeline behavior

1. `fetch_btc_data.py` can download Bitcoin data from yfinance and export as semicolon-delimited CSV.
2. `btc_data_processing.py` loads the yfinance CSV (semicolon-delimited) and parses Date/Datetime and OHLC columns.
3. It cleans the raw data, converts text fields to numeric, and drops invalid or duplicate rows.
3. Time-based features are added: day of week, month, week of year, and weekend flag.
4. Technical features are added: moving averages, volume averages, percent change, log returns,
   previous close/volume, spreads, and 14-day RSI.
5. A one-day-ahead target column `target_1d` is created.
6. `btc_modeling.py` builds the feature matrix, scales features, trains models, and reports evaluation metrics.

## How to run

Open a terminal in the project folder:

```powershell
cd "c:\Users\hp\Desktop\Group Lima\Group_Lima_BTC_Price_prediction"
```

Run the data processing script (example using the yfinance CSV):

```powershell
python btc_data_processing.py btc_yfinance_2015_2026.csv
```

Run the modeling pipeline:

```powershell
python btc_modeling.py btc_yfinance_2015_2026.csv
```

Run the tests:

```powershell
python test_btc_data_processing.py
python test_btc_modeling.py
```

## Next steps to improve the model

- Add more advanced technical indicators: MACD, Bollinger Bands, and additional momentum/volatility features.
- Experiment with stronger models: XGBoost, LightGBM, or LSTM/RNN for time-series forecasting.
- Add visualizations: predicted vs actual close prices, residual plots, and performance over time.
- Implement hyperparameter tuning and cross-validation workflows.
- Generate a comprehensive model comparison report and performance summary.
