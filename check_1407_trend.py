import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

ticker = '1407.T'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"--- {ticker} Price History (Monthly samples) ---")
df_monthly = df.resample('ME').last()
print(df_monthly[['Close']])

print(f"\n--- Lowest point ---")
min_idx = df['Low'].idxmin()
print(f"Date: {min_idx.strftime('%Y-%m-%d')}, Low: {df.loc[min_idx, 'Low']:.0f}")

print(f"\n--- Highest point recently ---")
max_idx = df.loc[min_idx:]['High'].idxmax()
print(f"Date: {max_idx.strftime('%Y-%m-%d')}, High: {df.loc[max_idx, 'High']:.0f}")

