import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import sys

ticker = sys.argv[1] if len(sys.argv) > 1 else '4443.T'

# 1. データの取得 (日足)
print(f"Fetching daily data for {ticker}, ^DJI, ^N225...")
df_target = yf.download(ticker, period='2y', progress=False)
df_dji = yf.download('^DJI', period='2y', progress=False)
df_n225 = yf.download('^N225', period='2y', progress=False)

for df in [df_target, df_dji, df_n225]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.ffill(inplace=True)

df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

df_n225['N225_Close'] = df_n225['Close']
df_n225['N225_Return'] = df_n225['Close'].pct_change()

df_target['Date'] = pd.to_datetime(df_target['Date']).dt.tz_localize(None)
df_dji['Date_JP'] = pd.to_datetime(df_dji['Date_JP']).dt.tz_localize(None)
df_n225['Date'] = pd.to_datetime(df_n225['Date']).dt.tz_localize(None)

df = pd.merge(df_target, df_dji[['Date_JP', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df['DJI_Return'] = df['DJI_Return'].fillna(0)
df = pd.merge(df, df_n225[['Date', 'N225_Return']], on='Date', how='left')
df['N225_Return'] = df['N225_Return'].fillna(0)

# 為替データ
try:
    df_fx = pd.read_csv('forex-data.csv')
    df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
    df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
    df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')
    df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(-1)
    df = pd.merge(df, df_fx[['Date', 'USD_JPY_Return']], on='Date', how='left')
    df['USD_JPY_Return'] = df['USD_JPY_Return'].fillna(0)
except FileNotFoundError:
    df['USD_JPY_Return'] = 0.0

print("Calculating features...")
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

base_features = [
    'Return', 'Vol_Change', 'RSI', 'MACD_Ratio', 'MACD_Hist_Ratio',
    'ATR_Ratio', 'BB_Width_Ratio', 'BB_Pos',
    'Upper_Shadow_Ratio', 'Lower_Shadow_Ratio',
    'Upper_Shadow_5d_MA', 'Lower_Shadow_5d_MA',
    'DJI_Return', 'N225_Return', 'USD_JPY_Return'
]

# 評価基準日: 2月末 (2026-02-27 を最終学習日として 3月の実績と比較する)
cutoff_date = pd.to_datetime('2026-02-27')
df_train = df[df['Date'] <= cutoff_date].copy()
df_test = df[df['Date'] > cutoff_date].copy()

if df_test.empty:
    print("No data available for March yet.")
    sys.exit()

for i in range(1, 6):
    df_train[f'Target_High_{i}d'] = (df_train['High'].shift(-i) - df_train['Open'].shift(-i)) / df_train['Open'].shift(-i)
    df_train[f'Target_Low_{i}d']  = (df_train['Low'].shift(-i) - df_train['Open'].shift(-i)) / df_train['Open'].shift(-i)
    df_train[f'Target_Open_Gap_{i}d'] = (df_train['Open'].shift(-i) - df_train['Close'].shift(-i+1)) / df_train['Close'].shift(-i+1)

df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_train_clean = df_train.dropna(subset=base_features + [f'Target_High_{i}d' for i in range(1,6)] + [f'Target_Low_{i}d' for i in range(1,6)])
X_train = df_train_clean[base_features]

models = {}
for i in range(1, 6):
    y_high = df_train_clean[f'Target_High_{i}d']
    model_high = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.90, n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42, tree_method='hist')
    model_high.fit(X_train, y_high)
    models[f'High_{i}d'] = model_high
    
    y_low = df_train_clean[f'Target_Low_{i}d']
    model_low = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.10, n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42, tree_method='hist')
    model_low.fit(X_train, y_low)
    models[f'Low_{i}d'] = model_low
    
    y_open_gap = df_train_clean[f'Target_Open_Gap_{i}d']
    model_open = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, tree_method='hist')
    model_open.fit(X_train, y_open_gap)
    models[f'Open_Gap_{i}d'] = model_open

# 2月末時点での予測
last_row = df_train.iloc[-1].copy()
last_row.fillna(0, inplace=True)
X_pred = pd.DataFrame([last_row[base_features]])

prev_close = last_row['Close']
print(f"\n=== 2/27(金)終値 {prev_close:.0f}円 を起点とした3月1週目の予測と実績の比較 ===")

predictions = []
actuals = df_test.head(5).reset_index(drop=True)

for i in range(1, min(6, len(actuals) + 1)):
    pred_open_gap = models[f'Open_Gap_{i}d'].predict(X_pred)[0]
    pred_open = prev_close * (1 + pred_open_gap)
    
    pred_high_ratio = max(models[f'High_{i}d'].predict(X_pred)[0], 0.0)
    pred_low_ratio = min(models[f'Low_{i}d'].predict(X_pred)[0], 0.0)
    
    pred_high = pred_open * (1 + pred_high_ratio)
    pred_low = pred_open * (1 + pred_low_ratio)
    
    actual_row = actuals.iloc[i-1]
    
    # 評価ロジック：実際のHigh/Lowが予測範囲内に収まっているか？
    # Lowのヒット: 実際のLowが予測Low付近にあるか
    # Highのヒット: 実際のHighが予測High付近にあるか
    
    predictions.append({
        'Date': actual_row['Date'].strftime('%m/%d'),
        'Actual_Low': actual_row['Low'],
        'Pred_Low': np.round(pred_low, 0),
        'Low_Error': np.round(actual_row['Low'] - pred_low, 0),
        'Actual_High': actual_row['High'],
        'Pred_High': np.round(pred_high, 0),
        'High_Error': np.round(actual_row['High'] - pred_high, 0)
    })
    
    prev_close = actual_row['Close'] # 翌日のOpen予測のために、実績のCloseを利用

df_res = pd.DataFrame(predictions)
print("-" * 80)
print(df_res.to_string(index=False))

# 評価のサマリー
mae_low = np.abs(df_res['Low_Error']).mean()
mae_high = np.abs(df_res['High_Error']).mean()

print(f"\n[Summary]")
print(f"安値(Low)の平均絶対誤差:  {mae_low:.1f} 円")
print(f"高値(High)の平均絶対誤差: {mae_high:.1f} 円")

