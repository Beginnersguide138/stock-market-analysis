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

# NOTE: For the main training path, this script trains on all available data and predicts
# the next unseen point (no train/test split). Technical indicators used here (Ichimoku,
# RSI, MACD, Stochastic, EMA) are backward-looking per row, so computing on the full
# dataset does not leak future OHLCV values into any given row's features. The CV section
# below recomputes features per fold to properly avoid leakage during evaluation.
df = compute_technical_features(df_sansan)

# 4. データ結合
df = pd.merge(df, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df['DJI_Return'] = df['DJI_Return'].fillna(0)

df = pd.merge(df, df_n225[['Date', 'N225_Close', 'N225_Return']], on='Date', how='left')
df['N225_Return'] = df['N225_Return'].fillna(0)

# 為替データ(USD/JPY)の統合（日次）
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(1)

df = pd.merge(df, df_fx[['Date', 'USD_JPY', 'USD_JPY_Return']], on='Date', how='left')
df['USD_JPY'].ffill(inplace=True)
df['USD_JPY_Return'].fillna(0, inplace=True)

# 5. 特徴量とターゲットの作成 (四本値予測)
# 翌日(1d)から5日後(5d)の四本値をターゲットにする
targets = ['Open', 'High', 'Low', 'Close']
for t in targets:
    for i in range(1, 6):
        df[f'Target_{t}_{i}d'] = df[t].shift(-i)

df_all = df.dropna().copy()
df_all = df_all.reset_index(drop=True)

features = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Vol_Change',
    'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB', 'Close_lag26',
    'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
    'DJI_Return', 'N225_Return', 'USD_JPY', 'USD_JPY_Return'
]

X = df_all[features]

# 6. 四本値の各モデルを学習
print("\nTraining models for OHLC (Open, High, Low, Close) over the next 5 days...")
models = {}

for t in targets:
    for i in range(1, 6):
        y = df_all[f'Target_{t}_{i}d']
        model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(X, y)
        models[f'{t}_{i}d'] = model

# 7. 直近のデータ（3/2実績）で来週の四本値を予測
latest_data = df.iloc[-1].copy()
latest_data = latest_data.fillna(0)
X_latest = pd.DataFrame([latest_data[features]])

predictions_ohlc = {
    'Date': [],
    'Open': [],
    'High': [],
    'Low': [],
    'Close': []
}

last_date = df['Date'].iloc[-1]
next_days = []
current_date = last_date
while len(next_days) < 5:
    current_date += timedelta(days=1)
    if current_date.weekday() < 5: # Monday to Friday
        next_days.append(current_date)

for d in next_days:
    predictions_ohlc['Date'].append(d.strftime('%Y-%m-%d'))

for i in range(1, 6):
    pred_open = models[f'Open_{i}d'].predict(X_latest)[0]
    pred_high = models[f'High_{i}d'].predict(X_latest)[0]
    pred_low = models[f'Low_{i}d'].predict(X_latest)[0]
    pred_close = models[f'Close_{i}d'].predict(X_latest)[0]
    
    # 論理的整合性（Low <= Open/Close <= High）を強制的に補正する
    actual_high = max(pred_high, pred_open, pred_close, pred_low)
    actual_low = min(pred_low, pred_open, pred_close, pred_high)
    
    predictions_ohlc['Open'].append(np.round(pred_open, 0))
    predictions_ohlc['High'].append(np.round(actual_high, 0))
    predictions_ohlc['Low'].append(np.round(actual_low, 0))
    predictions_ohlc['Close'].append(np.round(pred_close, 0))

pred_df = pd.DataFrame(predictions_ohlc)

print("\n=== 本日(3/2)以降のSansan (4443) 四本値(OHLC) 株価予測 ===")
print(f"基準日 (最新終値): {last_date.strftime('%Y-%m-%d')} (終値: {latest_data['Close']} 円)")
print("-------------------------------------------------")
print(pred_df.to_string(index=False))

# --- Cross Validation with TimeSeriesSplit ---
# NOTE: Features are recomputed per fold on training data only to avoid data leakage.
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n=== TimeSeriesSplit Cross-Validation (5-fold) ===")
tscv = TimeSeriesSplit(n_splits=5)

# Prepare raw data with external merges but without technical indicators for CV
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
            train_f[target_col] = train_f[t].shift(-i)
            train_f = train_f.dropna(subset=features + [target_col])
            skip = len(train_wb) - len(train_idx)
            if skip > 0:
                train_f = train_f.iloc[skip:]
            val_wb = df_raw_cv.iloc[buffer_start:val_idx[-1]+1]
            val_f = compute_technical_features(val_wb)
            val_f[target_col] = val_f[t].shift(-i)
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
