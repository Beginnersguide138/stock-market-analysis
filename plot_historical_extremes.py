import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
import xgboost as xgb
import plotly.graph_objects as go
from datetime import timedelta, datetime
import warnings
warnings.filterwarnings('ignore')
import sys

ticker = sys.argv[1] if len(sys.argv) > 1 else '4443.T'

# 1. データの取得 (日足) - 予測のために少し余裕をもたせて2年分取得し、学習時に1年に絞る
print(f"Fetching daily data for {ticker}, ^DJI, ^N225...")
df_target = yf.download(ticker, period='3y', progress=False)
df_dji = yf.download('^DJI', period='3y', progress=False)
df_n225 = yf.download('^N225', period='3y', progress=False)

for df in [df_target, df_dji, df_n225]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.ffill(inplace=True)

df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))
df_n225['N225_Close'] = df_n225['Close']
df_n225['N225_Return'] = df_n225['Close'].pct_change()

df_target['Date'] = pd.to_datetime(df_target['Date']).dt.tz_localize(None)
df_dji['Date_JP'] = pd.to_datetime(df_dji['Date_JP']).dt.tz_localize(None)
df_n225['Date'] = pd.to_datetime(df_n225['Date']).dt.tz_localize(None)

df = pd.merge(df_target, df_dji[['Date_JP', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df['DJI_Return'] = df['DJI_Return'].fillna(0)
df = pd.merge(df, df_n225[['Date', 'N225_Return']], on='Date', how='left')
df['N225_Return'] = df['N225_Return'].fillna(0)

print("Calculating features...")
df['Return'] = df['Close'].pct_change()
atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
df['ATR'] = atr.average_true_range()
df['ATR_Ratio'] = df['ATR'] / df['Close']
bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
df['BB_Width'] = bb.bollinger_wband()
df['BB_Width_Ratio'] = df['BB_Width'] / 100.0
df['BB_Pos'] = bb.bollinger_pband()

df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / df['Close']
df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / df['Close']
df['Upper_Shadow_5d_MA'] = df['Upper_Shadow_Ratio'].rolling(window=5).mean()
df['Lower_Shadow_5d_MA'] = df['Lower_Shadow_Ratio'].rolling(window=5).mean()

df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
macd = MACD(close=df['Close'])
df['MACD_Ratio'] = macd.macd() / df['Close']
df['MACD_Hist_Ratio'] = macd.macd_diff() / df['Close']
df['Vol_Change'] = df['Volume'].pct_change()

for i in range(1, 6):
    df[f'Target_High_{i}d'] = (df['High'].shift(-i) - df['Open'].shift(-i)) / df['Open'].shift(-i)
    df[f'Target_Low_{i}d']  = (df['Low'].shift(-i) - df['Open'].shift(-i)) / df['Open'].shift(-i)
    df[f'Target_Open_Gap_{i}d'] = (df['Open'].shift(-i) - df['Close'].shift(-i+1)) / df['Close'].shift(-i+1)

base_features = [
    'Return', 'Vol_Change', 'RSI', 'MACD_Ratio', 'MACD_Hist_Ratio',
    'ATR_Ratio', 'BB_Width_Ratio', 'BB_Pos',
    'Upper_Shadow_Ratio', 'Lower_Shadow_Ratio',
    'Upper_Shadow_5d_MA', 'Lower_Shadow_5d_MA',
    'DJI_Return', 'N225_Return'
]

df = df.replace([np.inf, -np.inf], np.nan)
df_clean = df.dropna(subset=base_features + [f'Target_High_1d', 'Target_Low_1d']).copy()

# シミュレーション期間: 2026年2月1日 〜 現在
start_eval = pd.to_datetime('2026-02-01')
end_eval = df_clean['Date'].max()

eval_dates = df_clean[(df_clean['Date'] >= start_eval) & (df_clean['Date'] <= end_eval)]['Date'].tolist()
print(f"\nRunning point-in-time predictions from {start_eval.strftime('%Y-%m-%d')} to {end_eval.strftime('%Y-%m-%d')}")

# 予測結果を格納するリスト (1日先の予測を繋ぎ合わせる)
historical_preds = []

# 評価日（各営業日）ごとに、その前日までの1年間のデータで学習し、翌日の高値・安値を予測する
for base_date in eval_dates:
    # 1年前の日付を計算
    cutoff_train_start = base_date - pd.Timedelta(days=365)
    
    # base_date よりも「前」のデータを学習に使う
    df_train = df_clean[(df_clean['Date'] >= cutoff_train_start) & (df_clean['Date'] < base_date)].copy()
    
    if len(df_train) < 50: # 学習データが少なすぎる場合はスキップ
        continue
        
    X_train = df_train[base_features]
    
    # 1日後の予測モデルのみ学習 (日々の軌跡を描画するため)
    y_high = df_train['Target_High_1d']
    model_high = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.90, n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, tree_method='hist', n_jobs=1)
    model_high.fit(X_train, y_high)
    
    y_low = df_train['Target_Low_1d']
    model_low = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.10, n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, tree_method='hist', n_jobs=1)
    model_low.fit(X_train, y_low)
    
    y_open_gap = df_train['Target_Open_Gap_1d']
    model_open = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, tree_method='hist', n_jobs=1)
    model_open.fit(X_train, y_open_gap)
    
    # 推論 (base_dateの前日のデータを使う)
    last_row = df_train.iloc[-1].copy()
    X_pred = pd.DataFrame([last_row[base_features]])
    prev_close = last_row['Close']
    
    pred_open_gap = model_open.predict(X_pred)[0]
    pred_open = prev_close * (1 + pred_open_gap)
    
    pred_high_ratio = max(model_high.predict(X_pred)[0], 0.0)
    pred_low_ratio = min(model_low.predict(X_pred)[0], 0.0)
    
    pred_high = pred_open * (1 + pred_high_ratio)
    pred_low = pred_open * (1 + pred_low_ratio)
    
    # 実際の実績値
    actual_row = df_clean[df_clean['Date'] == base_date].iloc[0]
    
    historical_preds.append({
        'Date': base_date,
        'Actual_Open': actual_row['Open'],
        'Actual_High': actual_row['High'],
        'Actual_Low': actual_row['Low'],
        'Actual_Close': actual_row['Close'],
        'Pred_Open': pred_open,
        'Pred_High': pred_high,
        'Pred_Low': pred_low
    })

df_res = pd.DataFrame(historical_preds)

print("Generating historical evaluation chart...")

fig = go.Figure()

# 実績のローソク足
fig.add_trace(go.Candlestick(
    x=df_res['Date'],
    open=df_res['Actual_Open'], high=df_res['Actual_High'],
    low=df_res['Actual_Low'], close=df_res['Actual_Close'],
    name='Actual (実績)',
    increasing_line_color='black', decreasing_line_color='black',
    increasing_fillcolor='white', decreasing_fillcolor='black'
))

# 予測のローソク足 (実体を持たせず、ヒゲだけのような表現にするため、Open=Closeとする)
# 少し右にずらして描画することもできるが、今回は同日の背景エラーバーとして描画する
fig.add_trace(go.Candlestick(
    x=df_res['Date'],
    open=df_res['Pred_Open'], high=df_res['Pred_High'],
    low=df_res['Pred_Low'], close=df_res['Pred_Open'],
    name='Predicted Extremes (予測範囲)',
    increasing_line_color='rgba(255, 0, 0, 0.5)', decreasing_line_color='rgba(0, 0, 255, 0.5)',
    increasing_fillcolor='rgba(255, 0, 0, 0.2)', decreasing_fillcolor='rgba(0, 0, 255, 0.2)'
))

# 予測の上限と下限を線でも結ぶ
fig.add_trace(go.Scatter(
    x=df_res['Date'], y=df_res['Pred_High'],
    mode='lines', line=dict(color='rgba(255, 0, 0, 0.3)', dash='dot'),
    name='Pred High (90%分位点)'
))
fig.add_trace(go.Scatter(
    x=df_res['Date'], y=df_res['Pred_Low'],
    mode='lines', line=dict(color='rgba(0, 0, 255, 0.3)', dash='dot'),
    name='Pred Low (10%分位点)'
))

fig.update_layout(
    title=f'【1年学習モデル】 {ticker} 2月〜現在の高値・安値 予測軌跡',
    yaxis_title='株価 (円)',
    xaxis_rangeslider_visible=False,
    height=800, width=1200, template='plotly_white',
    hovermode='x unified'
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_html = f"historical_extremes_{ticker.replace('.T','')}_{timestamp}.html"
fig.write_html(output_html)
print(f"\n=> チャートを {output_html} に保存しました。")
