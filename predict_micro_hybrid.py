import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
import xgboost as xgb
import plotly.graph_objects as go
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

# Note: ffill applied before split for simplicity; minimal leakage risk for forward-fill
df_sansan.ffill(inplace=True)
df_dji.ffill(inplace=True)
df_n225.ffill(inplace=True)

# 2. 指数・為替の特徴量作成
# ダウ (前日)
df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

# 日経平均 (同日)
df_n225['N225_Close'] = df_n225['Close']
df_n225['N225_Return'] = df_n225['Close'].pct_change()

# 3. テクニカル指標の計算 (Sansan)
def compute_technical_features(df_input):
    """Compute technical indicators on the given dataframe to avoid data leakage.
    Must be called on training data only (not on the full dataset before splitting)."""
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

# 4. データ結合 (external data merges - these don't cause leakage)
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx = df_fx.sort_values('Date') # 昇順ソートを追加
df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(1) # -1 (未来のカンニング) から 1 (過去の実績) へ修正

def merge_external_data(df_input):
    """Merge DJI, N225, and forex data into the given dataframe."""
    df_out = df_input.copy()
    df_out = pd.merge(df_out, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
    df_out['DJI_Return'] = df_out['DJI_Return'].fillna(0)
    df_out = pd.merge(df_out, df_n225[['Date', 'N225_Close', 'N225_Return']], on='Date', how='left')
    df_out['N225_Return'] = df_out['N225_Return'].fillna(0)
    df_out = pd.merge(df_out, df_fx[['Date', 'USD_JPY', 'USD_JPY_Return']], on='Date', how='left')
    df_out['USD_JPY'].ffill(inplace=True)
    df_out['USD_JPY_Return'].fillna(0, inplace=True)
    return df_out

features = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Vol_Change',
    'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB', 'Close_lag26',
    'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
    'DJI_Return', 'N225_Return', 'USD_JPY', 'USD_JPY_Return'
]

targets = ['Open', 'High', 'Low', 'Close']

# 5. Data leakage fix: compute features only on data up to the prediction date.
# Technical indicators (Ichimoku SpanA/B shift 26 periods forward) must not see future data.
test_date_start = pd.to_datetime('2026-03-03')
df_train_raw = df_sansan[df_sansan['Date'] < test_date_start].copy()
df_train = compute_technical_features(df_train_raw)
df_train = merge_external_data(df_train)

for t in targets:
    for i in range(1, 6):
        # 絶対価格ではなく「現在の終値からのリターン（乖離率）」をターゲットにする
        df_train[f'Target_{t}_{i}d'] = (df_train[t].shift(-i) - df_train['Close']) / df_train['Close']

X_train_full = df_train.dropna().reset_index(drop=True)

# 6. 四本値の各モデルを学習
print("\nTraining models for OHLC (Open, High, Low, Close) over the next 5 days...")
models = {}

for t in targets:
    for i in range(1, 6):
        y = X_train_full[f'Target_{t}_{i}d']
        model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(X_train_full[features], y)
        models[f'{t}_{i}d'] = model

# 7. Predict from the latest data point (3/2) using features computed only up to that date.
base_data = df_train.iloc[-1].copy()
base_data = base_data.fillna(0)
X_base = pd.DataFrame([base_data[features]])
current_close = base_data['Close']

predictions_ohlc = {
    'Date': [],
    'Open': [],
    'High': [],
    'Low': [],
    'Close': []
}

base_date = base_data['Date']
next_days = []
current_date = base_date
while len(next_days) < 5:
    current_date += timedelta(days=1)
    if current_date.weekday() < 5: # Monday to Friday
        next_days.append(current_date)

for d in next_days:
    predictions_ohlc['Date'].append(d.strftime('%Y-%m-%d'))

for i in range(1, 6):
    pred_open_return = models[f'Open_{i}d'].predict(X_base)[0]
    pred_high_return = models[f'High_{i}d'].predict(X_base)[0]
    pred_low_return = models[f'Low_{i}d'].predict(X_base)[0]
    pred_close_return = models[f'Close_{i}d'].predict(X_base)[0]
    
    # リターンから絶対価格に復元
    pred_open = current_close * (1 + pred_open_return)
    pred_high = current_close * (1 + pred_high_return)
    pred_low = current_close * (1 + pred_low_return)
    pred_close = current_close * (1 + pred_close_return)
    
    # 論理的整合性（Low <= Open/Close <= High）を強制的に補正する
    actual_high = max(pred_high, pred_open, pred_close, pred_low)
    actual_low = min(pred_low, pred_open, pred_close, pred_high)
    
    predictions_ohlc['Open'].append(np.round(pred_open, 0))
    predictions_ohlc['High'].append(np.round(actual_high, 0))
    predictions_ohlc['Low'].append(np.round(actual_low, 0))
    predictions_ohlc['Close'].append(np.round(pred_close, 0))

pred_df = pd.DataFrame(predictions_ohlc)

print("\n=== 本日(3/2)の実績を起点とした、明日(3/3〜3/9)のAI予測 ===")
print(f"基準日: {base_date.strftime('%Y-%m-%d')} (終値: {base_data['Close']} 円)")
print("-------------------------------------------------")
print(pred_df.to_string(index=False))

# --- Cross Validation with TimeSeriesSplit ---
# NOTE: Features are recomputed per fold on training data only to avoid data leakage.
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n=== TimeSeriesSplit Cross-Validation (5-fold) ===")
tscv = TimeSeriesSplit(n_splits=5)

df_raw_cv = df_sansan.copy()
df_raw_cv = pd.merge(df_raw_cv, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df_raw_cv['DJI_Return'] = df_raw_cv['DJI_Return'].fillna(0)
df_raw_cv = pd.merge(df_raw_cv, df_n225[['Date', 'N225_Close', 'N225_Return']], on='Date', how='left')
df_raw_cv['N225_Return'] = df_raw_cv['N225_Return'].fillna(0)
df_raw_cv = pd.merge(df_raw_cv, df_fx[['Date', 'USD_JPY', 'USD_JPY_Return']], on='Date', how='left')
df_raw_cv['USD_JPY'].ffill(inplace=True)
df_raw_cv['USD_JPY_Return'].fillna(0, inplace=True)
df_raw_cv = df_raw_cv.dropna(subset=['Close']).reset_index(drop=True)

BUFFER = 60
indices = np.arange(len(df_raw_cv))
for t in targets:
    for i in range(1, 6):
        target_col = f'Target_{t}_{i}d'
        mae_scores = []
        rmse_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(indices), 1):
            buffer_start = max(0, train_idx[0] - BUFFER)
            train_wb = df_raw_cv.iloc[buffer_start:train_idx[-1]+1]
            train_f = compute_technical_features(train_wb)
            train_f[target_col] = (train_f[t].shift(-i) - train_f['Close']) / train_f['Close']
            train_f = train_f.dropna(subset=features + [target_col])
            skip = len(train_wb) - len(train_idx)
            if skip > 0:
                train_f = train_f.iloc[skip:]
            val_wb = df_raw_cv.iloc[buffer_start:val_idx[-1]+1]
            val_f = compute_technical_features(val_wb)
            val_f[target_col] = (val_f[t].shift(-i) - val_f['Close']) / val_f['Close']
            val_f = val_f.dropna(subset=features + [target_col])
            val_f = val_f.iloc[-(len(val_idx)):]
            val_f = val_f.dropna(subset=features + [target_col])
            if len(val_f) == 0 or len(train_f) == 0:
                continue
            model_cv = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
            model_cv.fit(train_f[features], train_f[target_col])
            pred = model_cv.predict(val_f[features])
            mae_scores.append(mean_absolute_error(val_f[target_col], pred))
            rmse_scores.append(np.sqrt(mean_squared_error(val_f[target_col], pred)))
        print(f"  {target_col} Avg MAE: {np.mean(mae_scores):.4f}, Avg RMSE: {np.mean(rmse_scores):.4f}")
