import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from tqdm import tqdm
import sys

ticker = '4443.T'

# Get 10 years of data
df_target = yf.download(ticker, period='10y', progress=False)
df_dji = yf.download('^DJI', period='10y', progress=False)
df_n225 = yf.download('^N225', period='10y', progress=False)

for df_ in [df_target, df_dji, df_n225]:
    if isinstance(df_.columns, pd.MultiIndex):
        df_.columns = df_.columns.get_level_values(0)
    df_.reset_index(inplace=True)
    df_['Date'] = pd.to_datetime(df_['Date']).dt.tz_localize(None)

df_target = df_target.sort_values('Date').dropna(subset=['Close'])
df_dji = df_dji.sort_values('Date').dropna(subset=['Close'])
df_n225 = df_n225.sort_values('Date').dropna(subset=['Close'])

df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))
df_n225['N225_Close'] = df_n225['Close']
df_n225['N225_Return'] = df_n225['Close'].pct_change()

df_target['Date'] = pd.to_datetime(df_target['Date']).astype('datetime64[ns]')
df_dji['Date_JP'] = pd.to_datetime(df_dji['Date_JP']).astype('datetime64[ns]')
df_n225['Date'] = pd.to_datetime(df_n225['Date']).astype('datetime64[ns]')

df = pd.merge_asof(df_target, df_dji[['Date_JP', 'DJI_Return']], left_on='Date', right_on='Date_JP', direction='nearest', tolerance=pd.Timedelta(days=4))
df['DJI_Return'] = df['DJI_Return'].fillna(0)
df = pd.merge_asof(df, df_n225[['Date', 'N225_Return']], on='Date', direction='nearest', tolerance=pd.Timedelta(days=4))
df['N225_Return'] = df['N225_Return'].fillna(0)

df['Return'] = df['Close'].pct_change()
atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
df['ATR'] = atr.average_true_range()
df['ATR_Ratio'] = df['ATR'] / df['Close']
bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
df['BB_Width'] = bb.bollinger_wband()
df['BB_Width_Ratio'] = df['BB_Width'] / 100.0
df['BB_Pos'] = bb.bollinger_pband()
df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / df['Close']
df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / df['Close']
df['Upper_Shadow_5d_MA'] = df['Upper_Shadow_Ratio'].rolling(window=5).mean()
df['Lower_Shadow_5d_MA'] = df['Lower_Shadow_Ratio'].rolling(window=5).mean()
df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
macd = MACD(close=df['Close'])
df['MACD_Ratio'] = macd.macd() / df['Close']
df['MACD_Hist_Ratio'] = macd.macd_diff() / df['Close']
df['Vol_Change'] = df['Volume'].pct_change()

df = df.replace([np.inf, -np.inf], np.nan)

targets = ['Target_High_1d', 'Target_Low_1d', 'Target_Open_Gap_1d']
df['Target_High_1d'] = (df['High'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1)
df['Target_Low_1d']  = (df['Low'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1)
df['Target_Open_Gap_1d'] = (df['Open'].shift(-1) - df['Close']) / df['Close']

base_features = [
    'Return', 'Vol_Change', 'RSI', 'MACD_Ratio', 'MACD_Hist_Ratio',
    'ATR_Ratio', 'BB_Width_Ratio', 'BB_Pos',
    'Upper_Shadow_Ratio', 'Lower_Shadow_Ratio',
    'Upper_Shadow_5d_MA', 'Lower_Shadow_5d_MA',
    'DJI_Return', 'N225_Return'
]

df_clean = df.dropna(subset=base_features + targets).copy()

# Backtesting function
def run_backtest(df_data, lookback_years, start_date, end_date):
    eval_dates = df_data[(df_data['Date'] >= start_date) & (df_data['Date'] <= end_date)]['Date'].tolist()
    results = []
    
    for base_date in eval_dates:
        # Filter training data based on lookback
        if lookback_years is None:
            df_train = df_data[df_data['Date'] < base_date].copy()
        else:
            cutoff = base_date - pd.Timedelta(days=365 * lookback_years)
            df_train = df_data[(df_data['Date'] >= cutoff) & (df_data['Date'] < base_date)].copy()
            
        if len(df_train) < 50:
            continue
            
        X_train = df_train[base_features]
        
        # Train High
        model_high = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.90, n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, tree_method='hist', n_jobs=1)
        model_high.fit(X_train, df_train['Target_High_1d'])
        
        # Train Low
        model_low = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.10, n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, tree_method='hist', n_jobs=1)
        model_low.fit(X_train, df_train['Target_Low_1d'])
        
        # Train Open
        model_open = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, tree_method='hist', n_jobs=1)
        model_open.fit(X_train, df_train['Target_Open_Gap_1d'])
        
        # Predict for base_date
        # To predict base_date, we use features from base_date - 1 (the last row of df_train)
        last_row = df_train.iloc[-1]
        X_pred = pd.DataFrame([last_row[base_features]])
        prev_close = last_row['Close']
        
        pred_open_gap = model_open.predict(X_pred)[0]
        pred_open = prev_close * (1 + pred_open_gap)
        
        pred_high_ratio = max(model_high.predict(X_pred)[0], 0.0)
        pred_low_ratio = min(model_low.predict(X_pred)[0], 0.0)
        
        pred_high = pred_open * (1 + pred_high_ratio)
        pred_low = pred_open * (1 + pred_low_ratio)
        
        # Actuals
        actual_row = df_data[df_data['Date'] == base_date].iloc[0]
        
        results.append({
            'Date': base_date,
            'Actual_Low': actual_row['Low'],
            'Pred_Low': pred_low,
            'Low_Error': np.abs(actual_row['Low'] - pred_low),
            'Actual_High': actual_row['High'],
            'Pred_High': pred_high,
            'High_Error': np.abs(actual_row['High'] - pred_high)
        })
        
    return pd.DataFrame(results)

# Define evaluation period: Feb 1, 2026 to Mar 6, 2026
start_eval = pd.to_datetime('2026-02-01')
end_eval = pd.to_datetime('2026-03-06')

print("Evaluating 3-Year Lookback...")
res_3y = run_backtest(df_clean, lookback_years=3, start_date=start_eval, end_date=end_eval)

print("Evaluating 2-Year Lookback...")
res_2y = run_backtest(df_clean, lookback_years=2, start_date=start_eval, end_date=end_eval)

print("Evaluating 1-Year Lookback...")
res_1y = run_backtest(df_clean, lookback_years=1, start_date=start_eval, end_date=end_eval)

print("\n=== 2月〜3月の予測精度比較 (MAE: 平均絶対誤差) ===")
print(f"[直近3年間のみ学習]")
print(f"安値(Low)の誤差:  {res_3y['Low_Error'].mean():.1f} 円")
print(f"高値(High)の誤差: {res_3y['High_Error'].mean():.1f} 円")

print(f"\n[直近2年間のみ学習]")
print(f"安値(Low)の誤差:  {res_2y['Low_Error'].mean():.1f} 円")
print(f"高値(High)の誤差: {res_2y['High_Error'].mean():.1f} 円")

print(f"\n[直近1年間のみ学習]")
print(f"安値(Low)の誤差:  {res_1y['Low_Error'].mean():.1f} 円")
print(f"高値(High)の誤差: {res_1y['High_Error'].mean():.1f} 円")

