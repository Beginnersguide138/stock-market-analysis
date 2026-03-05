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
print("Fetching daily data for 4443.T and ^DJI from Yahoo Finance...")
df_sansan = yf.download('4443.T', period='2y')
df_dji = yf.download('^DJI', period='2y')

# yfinanceのMultiIndexをフラットにする処理
if isinstance(df_sansan.columns, pd.MultiIndex):
    df_sansan.columns = df_sansan.columns.get_level_values(0)
if isinstance(df_dji.columns, pd.MultiIndex):
    df_dji.columns = df_dji.columns.get_level_values(0)

df_sansan = df_sansan.reset_index()
df_dji = df_dji.reset_index()

# Note: ffill applied before split for simplicity; minimal leakage risk for forward-fill
df_sansan.ffill(inplace=True)
df_dji.ffill(inplace=True)

# 2. ダウ・ジョーンズの特徴量作成
# 日本市場が開く前にダウの終値が確定しているため、
# 日付tのSansanを予測する際、日付t-1のダウの変動を使用する
df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)

# 金曜日のダウは月曜日の日本市場に影響するため、曜日の調整
# もしDate_JPが土曜または日曜なら月曜日にする
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

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
    df_out['Return'] = df_out['Close'].pct_change()
    df_out['Vol_Change'] = df_out['Volume'].pct_change()
    return df_out

# NOTE: For the main training path, this script trains on all available data and predicts
# the next unseen point (no train/test split). Technical indicators used here (Ichimoku,
# RSI, MACD, Stochastic) are backward-looking per row, so computing on the full dataset
# does not leak future OHLCV values into any given row's features. The CV section below
# recomputes features per fold to properly avoid leakage during evaluation.
df = compute_technical_features(df_sansan)

# 4. データ結合
df = pd.merge(df, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df['DJI_Return'] = df['DJI_Return'].fillna(0) # 祝日等でデータがない場合は0

# 5. 特徴量とターゲットの作成
# 明日の終値を予測
df['Target_Close_1d'] = df['Close'].shift(-1)
df['Target_Close_2d'] = df['Close'].shift(-2)
df['Target_Close_3d'] = df['Close'].shift(-3)
df['Target_Close_4d'] = df['Close'].shift(-4)
df['Target_Close_5d'] = df['Close'].shift(-5)

# 欠損値を除去
df_train = df.dropna().copy()

features = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Vol_Change',
    'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB', 'Close_lag26',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
    'DJI_Return'
]

X = df_train[features]

# モデル学習（5日分の予測をそれぞれ作成）
models = {}
for i in range(1, 6):
    y = df_train[f'Target_Close_{i}d']
    model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    models[f'{i}d'] = model

print("\nModels trained. Generating predictions for the next week...")

# 直近の最新データを使って予測
latest_data = df.iloc[-1].copy()
# 欠損があるかもしれないので直近の有効な値で埋める
latest_data = latest_data.fillna(0)

X_latest = pd.DataFrame([latest_data[features]])
predictions = []
for i in range(1, 6):
    pred = models[f'{i}d'].predict(X_latest)[0]
    predictions.append(pred)

# 予測結果の出力
last_date = df['Date'].iloc[-1]
# 日本の営業日のみ抽出 (簡易的に平日)
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

print("\n=== 来週のSansan (4443) 株価予測 (日次ミクロ予測) ===")
print(f"基準日 (最新終値): {last_date.strftime('%Y-%m-%d')} ({latest_data['Close']} 円)")
print("-------------------------------------------------")
print(pred_df.to_string(index=False))

# 特徴量重要度 (翌日予測のモデル)
importance = models['1d'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
print("\n--- 株価予測に最も影響を与えた指標トップ5 (翌日予測) ---")
print(feature_importance_df.head(5).to_string(index=False))

# 一目均衡表などのステータス表示
print("\n--- テクニカル分析ステータス (直近) ---")
print(f"終値: {latest_data['Close']}円")
print(f"一目均衡表 転換線: {latest_data['Ichi_Tenkan']:.1f}円 / 基準線: {latest_data['Ichi_Kijun']:.1f}円")
print(f"一目均衡表 先行スパンA: {latest_data['Ichi_SpanA']:.1f}円 / 先行スパンB: {latest_data['Ichi_SpanB']:.1f}円")
print(f"RSI (14日): {latest_data['RSI']:.1f}%")
print(f"MACD: {latest_data['MACD']:.1f} (Signal: {latest_data['MACD_Signal']:.1f})")
print(f"Stochastic K: {latest_data['Stoch_K']:.1f}%")

# 可視化の作成
# 過去30日分のデータと予測をプロット
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

# 雲 (Span A, Span B)
fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Ichi_SpanA'], line=dict(color='rgba(0,0,0,0)'), showlegend=False))
fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Ichi_SpanB'], line=dict(color='rgba(0,0,0,0)'), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)', name='雲'))

# 予測データポイント
fig.add_trace(go.Scatter(x=next_days, y=predictions, mode='lines+markers', 
                         line=dict(color='purple', dash='dash', width=3), 
                         marker=dict(size=8, symbol='star'),
                         name='来週のAI予測値'))

fig.update_layout(title="Sansan (4443) テクニカル指標と来週の株価予測", 
                  yaxis_title="株価 (円)", 
                  xaxis_rangeslider_visible=False,
                  height=800, width=1200)

fig.write_html("micro_forecast.html")
print("\nVisualization saved to micro_forecast.html")

# --- Cross Validation with TimeSeriesSplit ---
# NOTE: Features are recomputed per fold on training data only to avoid data leakage.
# The raw OHLCV data (df_sansan + DJI merge) is split first, then indicators are computed.
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n=== TimeSeriesSplit Cross-Validation (5-fold) ===")
tscv = TimeSeriesSplit(n_splits=5)

# Prepare raw data with DJI merge but without technical indicators for CV
df_raw_cv = df_sansan.copy()
df_raw_cv = pd.merge(df_raw_cv, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df_raw_cv['DJI_Return'] = df_raw_cv['DJI_Return'].fillna(0)
df_raw_cv = df_raw_cv.dropna(subset=['Close']).reset_index(drop=True)

target_cols = [f'Target_Close_{i}d' for i in range(1, 6)]
for target_col_name in target_cols:
    i_days = int(target_col_name.split('_')[-1].replace('d', ''))
    mae_scores = []
    rmse_scores = []
    indices = np.arange(len(df_raw_cv))
    for fold, (train_idx, val_idx) in enumerate(tscv.split(indices), 1):
        # Recompute features on training portion only (with buffer for indicator warmup)
        BUFFER = 60  # enough for Ichimoku(52) and other indicators
        buffer_start = max(0, train_idx[0] - BUFFER)
        train_with_buffer = df_raw_cv.iloc[buffer_start:train_idx[-1]+1]
        train_featured = compute_technical_features(train_with_buffer)
        train_featured[target_col_name] = train_featured['Close'].shift(-i_days)
        train_featured = train_featured.dropna(subset=features + [target_col_name])
        # Only use rows that are in the actual train_idx range (drop buffer rows)
        actual_train_start = len(train_with_buffer) - len(train_idx)
        if actual_train_start > 0:
            train_featured = train_featured.iloc[actual_train_start:]
        X_t = train_featured[features]
        y_t = train_featured[target_col_name]

        # Recompute features for validation portion (use training + val data, take val rows)
        val_with_buffer = df_raw_cv.iloc[buffer_start:val_idx[-1]+1]
        val_featured = compute_technical_features(val_with_buffer)
        val_featured[target_col_name] = val_featured['Close'].shift(-i_days)
        val_featured = val_featured.dropna(subset=features + [target_col_name])
        # Only take rows corresponding to val_idx
        val_featured = val_featured.iloc[-(len(val_idx)):]
        val_featured = val_featured.dropna(subset=features + [target_col_name])
        if len(val_featured) == 0 or len(X_t) == 0:
            continue
        X_v = val_featured[features]
        y_v = val_featured[target_col_name]

        model_cv = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
        model_cv.fit(X_t, y_t)
        pred = model_cv.predict(X_v)
        mae_scores.append(mean_absolute_error(y_v, pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_v, pred)))
        print(f"  {target_col_name} Fold {fold}: MAE={mae_scores[-1]:.4f}, RMSE={rmse_scores[-1]:.4f}")
    print(f"  {target_col_name} Avg MAE: {np.mean(mae_scores):.4f}, Avg RMSE: {np.mean(rmse_scores):.4f}")
