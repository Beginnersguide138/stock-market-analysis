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

# Note: ffill deferred to after train/test split to prevent data leakage

# 2. 指数・為替の特徴量作成
df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

df_n225['N225_Close'] = df_n225['Close']
df_n225['N225_Return'] = df_n225['Close'].pct_change()

# 3. テクニカル指標の計算関数 (データリーク防止: 必要な範囲のみで計算)
def compute_features(df_input):
    """テクニカル指標を計算する。入力dfのコピーに対して計算を行う。"""
    df_out = df_input.copy()

    ichi = IchimokuIndicator(high=df_out['High'], low=df_out['Low'], window1=9, window2=26, window3=52)
    df_out['Ichi_Tenkan'] = ichi.ichimoku_conversion_line()
    df_out['Ichi_Kijun'] = ichi.ichimoku_base_line()
    df_out['Ichi_SpanA'] = ichi.ichimoku_a()
    df_out['Ichi_SpanB'] = ichi.ichimoku_b()
    df_out['Close_lag26'] = df_out['Close'].shift(26)

    df_out['RSI'] = RSIIndicator(close=df_out['Close'], window=14).rsi()
    macd = MACD(close=df_out['Close'])
    df_out['MACD'] = macd.macd()
    df_out['MACD_Signal'] = macd.macd_signal()
    df_out['MACD_Hist'] = macd.macd_diff()

    stoch = StochasticOscillator(high=df_out['High'], low=df_out['Low'], close=df_out['Close'], window=14, smooth_window=3)
    df_out['Stoch_K'] = stoch.stoch()
    df_out['Stoch_D'] = stoch.stoch_signal()

    df_out['EMA_12'] = df_out['Close'].ewm(span=12, adjust=False).mean()
    df_out['EMA_26'] = df_out['Close'].ewm(span=26, adjust=False).mean()

    df_out['Return'] = df_out['Close'].pct_change()
    df_out['Vol_Change'] = df_out['Volume'].pct_change()

    return df_out

# 4. データ結合 (外部データのmergeは生データに対して行う - テクニカル指標は分割後に計算)
df_raw = df_sansan.copy()

df_raw = pd.merge(df_raw, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df_raw['DJI_Return'] = df_raw['DJI_Return'].fillna(0)

df_raw = pd.merge(df_raw, df_n225[['Date', 'N225_Close', 'N225_Return']], on='Date', how='left')
df_raw['N225_Return'] = df_raw['N225_Return'].fillna(0)

# 為替データ(USD/JPY)
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(-1)

df_raw = pd.merge(df_raw, df_fx[['Date', 'USD_JPY', 'USD_JPY_Return']], on='Date', how='left')
# Note: USD_JPY ffill deferred to after train/test split
df_raw['USD_JPY_Return'].fillna(0, inplace=True)

# 5. 特徴量とターゲットの作成 (四本値予測)
targets = ['Open', 'High', 'Low', 'Close']
for t in targets:
    for i in range(1, 6):
        df_raw[f'Target_{t}_{i}d'] = df_raw[t].shift(-i)

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

# データリーク防止: 基準日 + 予測horizon分までの生データのみでテクニカル指標を計算
cutoff_date = pd.to_datetime(target_base_date_str) + timedelta(days=10)  # 5営業日分のバッファ
df_cutoff = df_raw[df_raw['Date'] <= cutoff_date].copy()
df_all = compute_features(df_cutoff)

# 学習用データは1/29までを使い、NaNを取り除く (1-day gap to prevent leakage)
gap = pd.Timedelta(days=1)
df_train = df_all[df_all['Date'] <= pd.to_datetime(target_base_date_str) - gap].copy()
df_train.ffill(inplace=True)
df_train = df_train.dropna().reset_index(drop=True)

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
