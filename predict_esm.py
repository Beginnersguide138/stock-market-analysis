import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("Fetching daily data for 4443.T and ^DJI from Yahoo Finance...")
df_sansan = yf.download('4443.T', period='1y') # 過去1年に短縮して直近の重みを増す
df_dji = yf.download('^DJI', period='1y')

if isinstance(df_sansan.columns, pd.MultiIndex):
    df_sansan.columns = df_sansan.columns.get_level_values(0)
if isinstance(df_dji.columns, pd.MultiIndex):
    df_dji.columns = df_dji.columns.get_level_values(0)

df_sansan = df_sansan.reset_index()
df_dji = df_dji.reset_index()
# Note: ffill applied before split for simplicity; minimal leakage risk for forward-fill
df_sansan.ffill(inplace=True)
df_dji.ffill(inplace=True)

df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

df = df_sansan.copy()

# モメンタムオシレータの計算
# NOTE: RSI, Stochastic, and ROC are computed on the full dataset here. Since this script
# uses Exponential Smoothing (not ML with train/test split) and applies oscillators only
# as a heuristic correction to the latest data point, the leakage risk is minimal.
# For a rigorous approach, compute these indicators only on data up to the prediction date.
df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
df['Stoch_K'] = stoch.stoch()
df['ROC'] = ROCIndicator(close=df['Close'], window=10).roc() # Rate of Change (Momentum)

df['Return'] = df['Close'].pct_change()
df = pd.merge(df, df_dji[['Date_JP', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df['DJI_Return'] = df['DJI_Return'].fillna(0)
df = df.dropna().reset_index(drop=True)

# 指数平滑化 (Exponential Smoothing) を使用して、直近のデータに強いウェイトを置く
# Holt-Winters (ES) 
# トレンドを持つ時系列データに対して、直近の観測値により大きな重みを与える
# ここでは「終値」そのものをESMで予測するベースラインとする
train_close = df['Close'].values

# トレンドあり、季節性なしのESMモデルを構築 (Holtの線形トレンド)
esm_model = ExponentialSmoothing(train_close, trend='add', seasonal=None, initialization_method="estimated").fit()
# 今後5日間のESMベース予測
esm_forecast = esm_model.forecast(5)

# 機械学習(XGBoost)による残差(あるいはリターン)予測の補正アプローチではなく、
# 今回はオシレータとESMの予測値を利用して複合的にアプローチします。
# ESMは単変量なので、これにオシレータ（RSI、Stochastic、ROC）の「買われすぎ/売られすぎ」状態を加味したハイブリッド予測を作成します。

# 現在のオシレータ状態
latest_rsi = df['RSI'].iloc[-1]
latest_stoch = df['Stoch_K'].iloc[-1]
latest_roc = df['ROC'].iloc[-1]
latest_close = df['Close'].iloc[-1]

print("\n=== 現在のモメンタム・オシレータ状態 ===")
print(f"RSI (14): {latest_rsi:.1f}%")
print(f"Stochastic K: {latest_stoch:.1f}%")
print(f"Momentum (ROC 10): {latest_roc:.1f}%")

# オシレータベースの補正係数を算出 (非常にシンプルなロジック)
# RSIが低い(売られすぎ)場合は、ESMのトレンドより上にブレやすい
# RSIが高い(買われすぎ)場合は、ESMのトレンドより下にブレやすい
rsi_correction = (50 - latest_rsi) / 100.0  # RSI 40なら +0.10
stoch_correction = (50 - latest_stoch) / 100.0

# モメンタム(ROC)が下落トレンドにあるか、上昇トレンドにあるかをトレンドの勢いとして加算
momentum_factor = latest_roc / 100.0

# 予測日数のリスト作成
last_date = df['Date'].iloc[-1]
next_days = []
current_date = last_date
while len(next_days) < 5:
    current_date += timedelta(days=1)
    if current_date.weekday() < 5: # 平日
        next_days.append(current_date)

final_predictions = []
current_price = latest_close

for i in range(5):
    # ESMのベース予測値
    base_pred = esm_forecast[i]
    
    # 補正係数の適用 (日数が経つごとにオシレータの効力は減衰させ、モメンタムに依存させる)
    decay = 0.8 ** i
    adjustment = (current_price * rsi_correction * decay * 0.5) + (current_price * stoch_correction * decay * 0.5)
    
    # 最終予測値 = ベース予測 + オシレータ反発補正 + モメンタム持続力
    # ESMがすでにトレンドを内包しているため、オシレータによる「逆張り圧力」を足すイメージ
    pred = base_pred + adjustment
    
    final_predictions.append(pred)
    current_price = pred # 次の日の計算用

pred_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in next_days],
    'ESM_Base_Forecast': np.round(esm_forecast, 0),
    'Oscillator_Adjusted_Forecast': np.round(final_predictions, 0)
})

print("\n=== 来週のSansan (4443) 株価予測 (ESM + モメンタム/オシレータ補正) ===")
print(f"基準日 (最新終値): {last_date.strftime('%Y-%m-%d')} ({latest_close} 円)")
print("-------------------------------------------------")
print(pred_df.to_string(index=False))

