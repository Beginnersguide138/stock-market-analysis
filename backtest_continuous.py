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

# 4. データ結合 (外部データのmergeは生データに対して行う - テクニカル指標はループ内で計算)
df_raw = df_sansan.copy()

df_raw = pd.merge(df_raw, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df_raw['DJI_Return'] = df_raw['DJI_Return'].fillna(0)

df_raw = pd.merge(df_raw, df_n225[['Date', 'N225_Close', 'N225_Return']], on='Date', how='left')
df_raw['N225_Return'] = df_raw['N225_Return'].fillna(0)

df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(-1)

df_raw = pd.merge(df_raw, df_fx[['Date', 'USD_JPY', 'USD_JPY_Return']], on='Date', how='left')
# Note: USD_JPY ffill deferred to after train/test split
df_raw['USD_JPY_Return'].fillna(0, inplace=True)

# 5. 特徴量とターゲットの作成
targets = ['Close'] # 方向性だけを見るためCloseのみでOK
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

# === 連続バックテスト: 2/2 〜 2/20 の各週末(金曜)を起点に検証 ===
test_base_dates = ['2026-01-30', '2026-02-06', '2026-02-13', '2026-02-20']

total_correct = 0
total_predictions = 0

print("\n=== 波形（上下方向）の連続バックテスト検証 ===")

for base_date_str in test_base_dates:
    base_date_dt = pd.to_datetime(base_date_str)

    # データリーク防止: 基準日 + 予測horizon分までの生データのみでテクニカル指標を計算
    cutoff_date = base_date_dt + timedelta(days=10)  # 5営業日分のバッファ
    df_cutoff = df_raw[df_raw['Date'] <= cutoff_date].copy()
    df_iter = compute_features(df_cutoff)

    # 基準日までのデータで学習 (1-day gap to prevent leakage)
    gap = pd.Timedelta(days=1)
    df_train = df_iter[df_iter['Date'] <= base_date_dt - gap].copy()
    df_train.ffill(inplace=True)
    df_train = df_train.dropna().reset_index(drop=True)

    models = {}
    for i in range(1, 6):
        y = df_train[f'Target_Close_{i}d']
        model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(df_train[features], y)
        models[f'{i}d'] = model

    # 基準日時点のデータを取得
    base_data = df_iter[df_iter['Date'] <= base_date_dt].iloc[-1].copy()
    base_data.ffill(inplace=True)
    base_data = base_data.fillna(0)
    X_base = pd.DataFrame([base_data[features]])
    
    # 翌5日間の日付を計算
    base_date = base_data['Date']
    next_days = []
    current_date = base_date
    while len(next_days) < 5:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            next_days.append(current_date)
            
    # 予測
    pred_closes = []
    prev_close_pred = base_data['Close']
    directions_pred = []
    
    for i in range(1, 6):
        pred_c = models[f'{i}d'].predict(X_base)[0]
        pred_closes.append(pred_c)
        direction = "UP" if pred_c > prev_close_pred else "DOWN"
        directions_pred.append(direction)
        prev_close_pred = pred_c
        
    print(f"\n--- 起点: {base_date_str} (終値: {base_data['Close']}円) ---")
    
    # 実績との照合
    correct_in_batch = 0
    valid_days = 0
    prev_actual_close = base_data['Close']
    
    for i, d in enumerate(next_days):
        d_str = d.strftime('%Y-%m-%d')
        actual = df_iter[df_iter['Date'] == d_str]
        
        if not actual.empty:
            a_close = actual['Close'].values[0]
            actual_dir = "UP" if a_close > prev_actual_close else "DOWN"
            pred_dir = directions_pred[i]
            
            is_correct = "O" if actual_dir == pred_dir else "X"
            if is_correct == "O":
                correct_in_batch += 1
            valid_days += 1
            
            print(f"[{d_str}] 予測: {pred_dir} ({pred_closes[i]:.0f}円) | 実績: {actual_dir} ({a_close:.0f}円) -> {is_correct}")
            prev_actual_close = a_close
            
    if valid_days > 0:
        print(f"  -> 正答率: {correct_in_batch}/{valid_days} ({correct_in_batch/valid_days*100:.1f}%)")
        total_correct += correct_in_batch
        total_predictions += valid_days

print(f"\n=============================================")
print(f"総合正答率: {total_correct}/{total_predictions} ({total_correct/total_predictions*100:.1f}%)")
print(f"=============================================")