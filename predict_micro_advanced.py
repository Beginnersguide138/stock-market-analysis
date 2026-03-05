import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(-1) # 降順データの場合

df = pd.merge(df, df_fx[['Date', 'USD_JPY', 'USD_JPY_Return']], on='Date', how='left')
df['USD_JPY'].ffill(inplace=True)
df['USD_JPY_Return'].fillna(0, inplace=True)

# 5. 特徴量とターゲットの作成
df['Target_Close_1d'] = df['Close'].shift(-1)
df_train = df.dropna().copy()
df_train = df_train.reset_index(drop=True)

features = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Vol_Change',
    'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB', 'Close_lag26',
    'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
    'DJI_Return', 'N225_Return', 'USD_JPY', 'USD_JPY_Return'
]

X = df_train[features]
y = df_train['Target_Close_1d']

# 6. TimeSeriesSplit (時系列交差検証)
# NOTE: Features are recomputed per fold on training data only to avoid data leakage.
tscv = TimeSeriesSplit(n_splits=5)
print("\n=== 時系列K-Fold交差検証 (TimeSeriesSplit) の実行 ===")

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

BUFFER = 60  # enough for Ichimoku(52) and other indicators
mae_scores = []
rmse_scores = []
indices = np.arange(len(df_raw_cv))

for fold, (train_index, test_index) in enumerate(tscv.split(indices)):
    buffer_start = max(0, train_index[0] - BUFFER)

    # Recompute features on training portion (with buffer for indicator warmup)
    train_with_buffer = df_raw_cv.iloc[buffer_start:train_index[-1]+1]
    train_featured = compute_technical_features(train_with_buffer)
    train_featured['Target_Close_1d'] = train_featured['Close'].shift(-1)
    train_featured = train_featured.dropna(subset=features + ['Target_Close_1d'])
    actual_train_start = len(train_with_buffer) - len(train_index)
    if actual_train_start > 0:
        train_featured = train_featured.iloc[actual_train_start:]
    X_train_fold = train_featured[features]
    y_train_fold = train_featured['Target_Close_1d']

    # Recompute features for validation (use train + val data, take val rows)
    val_with_buffer = df_raw_cv.iloc[buffer_start:test_index[-1]+1]
    val_featured = compute_technical_features(val_with_buffer)
    val_featured['Target_Close_1d'] = val_featured['Close'].shift(-1)
    val_featured = val_featured.dropna(subset=features + ['Target_Close_1d'])
    val_featured = val_featured.iloc[-(len(test_index)):]
    val_featured = val_featured.dropna(subset=features + ['Target_Close_1d'])
    if len(val_featured) == 0 or len(X_train_fold) == 0:
        continue
    X_test_fold = val_featured[features]
    y_test_fold = val_featured['Target_Close_1d']

    model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train_fold, y_train_fold)

    preds = model.predict(X_test_fold)
    mae = mean_absolute_error(y_test_fold, preds)
    rmse = np.sqrt(mean_squared_error(y_test_fold, preds))

    mae_scores.append(mae)
    rmse_scores.append(rmse)

    print(f"Fold {fold+1}: MAE = {mae:.2f}円, RMSE = {rmse:.2f}円 (Train size: {len(X_train_fold)}, Test size: {len(X_test_fold)})")

print(f"\nAverage MAE across folds: {np.mean(mae_scores):.2f}円")
print(f"Average RMSE across folds: {np.mean(rmse_scores):.2f}円")

# 7. 全データで再学習して来週を予測
print("\nTraining final model on full dataset...")
# 翌日(1d)から5日後(5d)までそれぞれのモデルを学習
models = {}
for i in range(1, 6):
    df[f'Target_Close_{i}d'] = df['Close'].shift(-i)

df_final_train = df.dropna().copy()
X_final = df_final_train[features]

for i in range(1, 6):
    y_final = df_final_train[f'Target_Close_{i}d']
    model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_final, y_final)
    models[f'{i}d'] = model

# 直近の最新データを使って予測
latest_data = df.iloc[-1].copy()
latest_data = latest_data.fillna(0)

X_latest = pd.DataFrame([latest_data[features]])
predictions = []
for i in range(1, 6):
    pred = models[f'{i}d'].predict(X_latest)[0]
    predictions.append(pred)

last_date = df['Date'].iloc[-1]
next_days = []
current_date = last_date
while len(next_days) < 5:
    current_date += timedelta(days=1)
    if current_date.weekday() < 5: # Monday to Friday
        next_days.append(current_date)

pred_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in next_days],
    'Predicted_Close': np.round(predictions, 0)
})

print("\n=== 来週のSansan (4443) 株価予測 (XGBoost + 指数/為替特徴量) ===")
print(f"基準日 (最新終値): {last_date.strftime('%Y-%m-%d')} ({latest_data['Close']} 円)")
print("-------------------------------------------------")
print(pred_df.to_string(index=False))

# 特徴量重要度 (翌日予測のモデル)
importance = models['1d'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
print("\n--- 株価予測に最も影響を与えた指標トップ10 (翌日予測) ---")
print(feature_importance_df.head(10).to_string(index=False))

# 可視化の作成
df_plot = df.tail(30).copy()

fig = go.Figure()

# ローソク足
fig.add_trace(go.Candlestick(x=df_plot['Date'],
                open=df_plot['Open'], high=df_plot['High'],
                low=df_plot['Low'], close=df_plot['Close'],
                name='ローソク足'))

# 一目均衡表
fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Ichi_Tenkan'], line=dict(color='red', width=1), name='転換線'))
fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Ichi_Kijun'], line=dict(color='blue', width=1), name='基準線'))
fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Ichi_SpanA'], line=dict(color='rgba(0,0,0,0)'), showlegend=False))
fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Ichi_SpanB'], line=dict(color='rgba(0,0,0,0)'), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)', name='雲'))

# 予測データポイント
fig.add_trace(go.Scatter(x=next_days, y=predictions, mode='lines+markers', 
                         line=dict(color='purple', dash='dash', width=3), 
                         marker=dict(size=8, symbol='star'),
                         name='来週のAI予測値'))

fig.update_layout(title="Sansan (4443) テクニカル・指数連動型予測", 
                  yaxis_title="株価 (円)", 
                  xaxis_rangeslider_visible=False,
                  height=800, width=1200)

fig.write_html("advanced_micro_forecast.html")
print("\nVisualization saved to advanced_micro_forecast.html")
