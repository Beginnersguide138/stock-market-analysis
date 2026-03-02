import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
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
# ダウ (前日)
df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

# 日経平均 (同日)
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

# 為替データ(USD/JPY)の統合（日次）
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(-1)

df = pd.merge(df, df_fx[['Date', 'USD_JPY', 'USD_JPY_Return']], on='Date', how='left')
df['USD_JPY'].ffill(inplace=True)
df['USD_JPY_Return'].fillna(0, inplace=True)

# 5. 特徴量とターゲットの作成
# 翌日(1d)と翌々日(2d)を予測ターゲットにする
df['Target_Close_1d'] = df['Close'].shift(-1)
df['Target_Close_2d'] = df['Close'].shift(-2)

df_all = df.dropna().copy()
df_all = df_all.reset_index(drop=True)

features = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Vol_Change',
    'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB', 'Close_lag26',
    'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
    'DJI_Return', 'N225_Return', 'USD_JPY', 'USD_JPY_Return'
]

# === バックテスト: 2月末の2日間（2/26, 2/27）を予測できるか？ ===
# 2/25（水）までのデータを訓練データとする
test_date_start = pd.to_datetime('2026-02-26')
df_train = df_all[df_all['Date'] < test_date_start]

# 2/25時点の最新データ（予測の起点）
if df_train.empty:
    print("Error: 訓練データが空です。")
    exit()

base_data = df_train.iloc[-1].copy()
base_date = base_data['Date'].strftime('%Y-%m-%d')
base_close = base_data['Close']
print(f"\n=== バックテスト実行 (基準日: {base_date}, 終値: {base_close} 円) ===")
X_train = df_train[features]

# モデル学習
model_1d = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
model_2d = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)

model_1d.fit(X_train, df_train['Target_Close_1d'])
model_2d.fit(X_train, df_train['Target_Close_2d'])

# 2/25のデータを使って2/26と2/27を予測
X_test = pd.DataFrame([base_data[features]])
pred_26 = model_1d.predict(X_test)[0]
pred_27 = model_2d.predict(X_test)[0]

# 実際の正解データを取得
act_26_val = 1166.0
act_27_val = 1174.0

print("\n【2月26日 (木) の予測】")
print(f"  予測値: {pred_26:.0f} 円")
print(f"  実績値: {act_26_val} 円")
if act_26_val:
    print(f"  誤差: {abs(pred_26 - act_26_val):.0f} 円 ({abs(pred_26 - act_26_val)/act_26_val*100:.1f}%)")

print("\n【2月27日 (金) の予測】")
print(f"  予測値: {pred_27:.0f} 円")
print(f"  実績値: {act_27_val} 円")
if act_27_val:
    print(f"  誤差: {abs(pred_27 - act_27_val):.0f} 円 ({abs(pred_27 - act_27_val)/act_27_val*100:.1f}%)")
