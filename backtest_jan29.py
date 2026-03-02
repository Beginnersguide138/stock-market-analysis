import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
import xgboost as xgb
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# 1. データの取得
print("Fetching daily data for 4443.T (Sansan), ^DJI (Dow Jones), and ^N225 (Nikkei 225) from Yahoo Finance...")
df_sansan = yf.download('4443.T', period='2y')
df_dji = yf.download('^DJI', period='2y')
df_n225 = yf.download('^N225', period='2y')

if isinstance(df_sansan.columns, pd.MultiIndex):
    df_sansan.columns = df_sansan.columns.get_level_values(0)
if isinstance(df_dji.columns, pd.MultiIndex):
    df_dji.columns = df_dji.columns.get_level_values(0)
if isinstance(df_n225.columns, pd.MultiIndex):
    df_n225.columns = df_n225.columns.get_level_values(0)

df_sansan = df_sansan.reset_index()
df_dji = df_dji.reset_index()
df_n225 = df_n225.reset_index()

df_sansan.ffill(inplace=True)
df_dji.ffill(inplace=True)
df_n225.ffill(inplace=True)

# 2. 指数・為替の特徴量作成
df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

df_n225['N225_Close'] = df_n225['Close']
df_n225['N225_Return'] = df_n225['Close'].pct_change()

# 3. テクニカル指標の計算 (Sansan)
df = df_sansan.copy()

ichi = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
df['Ichi_Tenkan'] = ichi.ichimoku_conversion_line()
df['Ichi_Kijun'] = ichi.ichimoku_base_line()
df['Ichi_SpanA'] = ichi.ichimoku_a()
df['Ichi_SpanB'] = ichi.ichimoku_b()
df['Close_lag26'] = df['Close'].shift(26)

df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
macd = MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Hist'] = macd.macd_diff()

stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
df['Stoch_K'] = stoch.stoch()
df['Stoch_D'] = stoch.stoch_signal()

df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

df['Return'] = df['Close'].pct_change()
df['Vol_Change'] = df['Volume'].pct_change()

# 4. データ結合
df = pd.merge(df, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df['DJI_Return'] = df['DJI_Return'].fillna(0)

df = pd.merge(df, df_n225[['Date', 'N225_Close', 'N225_Return']], on='Date', how='left')
df['N225_Return'] = df['N225_Return'].fillna(0)

# 為替データ(USD/JPY)
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(-1)

df = pd.merge(df, df_fx[['Date', 'USD_JPY', 'USD_JPY_Return']], on='Date', how='left')
df['USD_JPY'].ffill(inplace=True)
df['USD_JPY_Return'].fillna(0, inplace=True)

# 5. 特徴量とターゲットの作成 (四本値予測)
targets = ['Open', 'High', 'Low', 'Close']
for t in targets:
    for i in range(1, 6):
        df[f'Target_{t}_{i}d'] = df[t].shift(-i)

df_all = df.copy() # dropnaしない

features = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Vol_Change',
    'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB', 'Close_lag26',
    'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
    'DJI_Return', 'N225_Return', 'USD_JPY', 'USD_JPY_Return'
]

# === バックテスト: 1/29（木）までのデータで、1/30以降の5日間を予測 ===
target_base_date_str = '2026-01-29'
test_date_start = pd.to_datetime('2026-01-30')

# 学習用データは1/29より前の確定期を使い、NaNを取り除く
df_train = df_all[df_all['Date'] < test_date_start].dropna().reset_index(drop=True)

print(f"\nTraining models for OHLC using data up to {target_base_date_str}...")
models = {}

for t in targets:
    for i in range(1, 6):
        y = df_train[f'Target_{t}_{i}d']
        model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(df_train[features], y)
        models[f'{t}_{i}d'] = model

# 1/29のデータを起点として予測
df_base = df_all[df_all['Date'] <= pd.to_datetime(target_base_date_str)]
if df_base.empty:
    print(f"Error: {target_base_date_str}のデータが見つかりません。")
    exit()

base_data = df_base.iloc[-1].copy()
base_data.ffill(inplace=True)
base_data = base_data.fillna(0)
X_base = pd.DataFrame([base_data[features]])

predictions_ohlc = {
    'Date': [],
    'Open': [],
    'High': [],
    'Low': [],
    'Close': [],
    'Direction_Pred': [] # 上下方向（前日終値比）
}

base_date = base_data['Date']
next_days = []
current_date = base_date
while len(next_days) < 5:
    current_date += timedelta(days=1)
    if current_date.weekday() < 5: # Monday to Friday
        next_days.append(current_date)

prev_close = base_data['Close']
for i, d in enumerate(next_days):
    predictions_ohlc['Date'].append(d.strftime('%Y-%m-%d'))

for i in range(1, 6):
    pred_open = models[f'Open_{i}d'].predict(X_base)[0]
    pred_high = models[f'High_{i}d'].predict(X_base)[0]
    pred_low = models[f'Low_{i}d'].predict(X_base)[0]
    pred_close = models[f'Close_{i}d'].predict(X_base)[0]
    
    actual_high = max(pred_high, pred_open, pred_close, pred_low)
    actual_low = min(pred_low, pred_open, pred_close, pred_high)
    
    predictions_ohlc['Open'].append(np.round(pred_open, 0))
    predictions_ohlc['High'].append(np.round(actual_high, 0))
    predictions_ohlc['Low'].append(np.round(actual_low, 0))
    predictions_ohlc['Close'].append(np.round(pred_close, 0))
    
    direction = "UP" if pred_close > prev_close else "DOWN"
    predictions_ohlc['Direction_Pred'].append(direction)
    prev_close = pred_close

pred_df = pd.DataFrame(predictions_ohlc)

print(f"\n=== {target_base_date_str} を起点とした、翌5日間のAI予測 ===")
print(f"基準日: {base_date.strftime('%Y-%m-%d')} (終値: {base_data['Close']} 円)")
print("-------------------------------------------------")
print(pred_df[['Date', 'Open', 'High', 'Low', 'Close', 'Direction_Pred']].to_string(index=False))

# 波形（方向性）の検証
print("\n--- 実際のデータとの波形（上下方向）検証 ---")
correct_directions = 0

prev_actual_close = base_data['Close']
for i, d_str in enumerate(pred_df['Date']):
    actual = df_all[df_all['Date'] == d_str]
    if not actual.empty:
        a_close = actual['Close'].values[0]
        actual_direction = "UP" if a_close > prev_actual_close else "DOWN"
        
        pred_direction = pred_df['Direction_Pred'].iloc[i]
        is_correct = "O" if actual_direction == pred_direction else "X"
        if is_correct == "O":
            correct_directions += 1
            
        print(f"[{d_str}] 予測: {pred_direction} ({pred_df['Close'].iloc[i]:.0f}円) | 実績: {actual_direction} ({a_close:.0f}円) -> 判定: {is_correct}")
        prev_actual_close = a_close
    else:
        print(f"[{d_str}] 実績データなし")

print(f"\n方向性の正答率: {correct_directions}/5 ({correct_directions/5*100:.1f}%)")
