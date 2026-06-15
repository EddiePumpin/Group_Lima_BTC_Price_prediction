import yfinance as yf
import pandas as pd

print("Downloading BTC data...")
df = yf.download("BTC-USD", start="2015-01-01", end="2026-05-11", interval="1d")
df = df.reset_index()

df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

df = df.rename(columns={
    "Date": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
})

df["timeOpen"] = df["timestamp"]
df["timeClose"] = df["timestamp"]
df["timeHigh"] = df["timestamp"]
df["timeLow"] = df["timestamp"]
df["marketCap"] = None
df["circulatingSupply"] = None
df["name"] = "BTC"

df.to_csv("btc_yfinance_2015_2026.csv", sep=";", index=False)
print(f"Done. Saved {len(df)} rows to btc_yfinance_2015_2026.csv")