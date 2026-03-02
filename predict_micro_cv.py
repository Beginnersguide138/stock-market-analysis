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
print("Fetching daily data for 4443.T and ^DJI from Yahoo Finance...")
df_sansan = yf.download('4443.T', period='2y')
df_dji = yf.download('^DJI', period='2y')

if isinstance(df_sansan.columns, pd.MultiIndex):
    df_sansan.columns = df_sansan.columns.get_level_values(0)
if isinstance(df_dji.columns, pd.MultiIndex):
    df_dji.columns = df_dji.columns.get_level_values(0)

df_sansan = df_sansan.reset_index()
df_dji = df_dji.reset_index()

df_sansan.ffill(inplace=True)
df_dji.ffill(inplace=True)

# 2. ダウ・ジョーンズの特徴量作成
df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

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

df['Return'] = df['Close'].pct_change()
df['Vol_Change'] = df['Volume'].pct_change()

# 4. データ結合
df = pd.merge(df, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df['DJI_Return'] = df['DJI_Return'].fillna(0)

# 5. 特徴量とターゲットの作成
df['Target_Close_1d'] = df['Close'].shift(-1)
df_train = df.dropna().copy()
df_train = df_train.reset_index(drop=True)

features = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Vol_Change',
    'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB', 'Close_lag26',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
    'DJI_Return'
]

X = df_train[features]
y = df_train['Target_Close_1d']

# 6. TimeSeriesSplit (時系列交差検証)
tscv = TimeSeriesSplit(n_splits=5)
print("\n=== 時系列K-Fold交差検証 (TimeSeriesSplit) の実行 ===")

mae_scores = []
rmse_scores = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
    # 訓練
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train_fold, y_train_fold)
    
    # 予測と評価
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
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
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

print("\n=== 来週のSansan (4443) 株価予測 (Cross-Validation検証後) ===")
print(f"基準日 (最新終値): {last_date.strftime('%Y-%m-%d')} ({latest_data['Close']} 円)")
print("-------------------------------------------------")
print(pred_df.to_string(index=False))
