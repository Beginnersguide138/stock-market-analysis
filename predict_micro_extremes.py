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
from joblib import Parallel, delayed
import sys

ticker = sys.argv[1] if len(sys.argv) > 1 else '4443.T'

# 1. データの取得 (日足)
print(f"Fetching daily data for {ticker}, ^DJI, ^N225...")
df_target = yf.download(ticker, period='1y', progress=False)
df_dji = yf.download('^DJI', period='1y', progress=False)
df_n225 = yf.download('^N225', period='1y', progress=False)

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

# 日付ベースの近似マージ（日足なので基本は完全一致に近い）
df_target['Date'] = pd.to_datetime(df_target['Date']).dt.tz_localize(None)
df_dji['Date_JP'] = pd.to_datetime(df_dji['Date_JP']).dt.tz_localize(None)
df_n225['Date'] = pd.to_datetime(df_n225['Date']).dt.tz_localize(None)

df = pd.merge(df_target, df_dji[['Date_JP', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df['DJI_Return'] = df['DJI_Return'].fillna(0)
df = pd.merge(df, df_n225[['Date', 'N225_Return']], on='Date', how='left')
df['N225_Return'] = df['N225_Return'].fillna(0)

# 為替データ
try:
    df_fx = pd.read_csv('forex-data.csv')
    df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
    df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
    df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')
    df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(-1)
    df = pd.merge(df, df_fx[['Date', 'USD_JPY_Return']], on='Date', how='left')
    df['USD_JPY_Return'] = df['USD_JPY_Return'].fillna(0)
except FileNotFoundError:
    df['USD_JPY_Return'] = 0.0

# 2. 特徴量の計算 (ボラティリティとヒゲ特化)
print("Calculating advanced micro features...")
df['Return'] = df['Close'].pct_change()

# ATR (Average True Range) -> その銘柄の「1日の真の変動幅」
atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
df['ATR'] = atr.average_true_range()
df['ATR_Ratio'] = df['ATR'] / df['Close'] # 現在の株価に対するボラティリティの割合

# ボリンジャーバンド
bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
df['BB_High'] = bb.bollinger_hband()
df['BB_Low'] = bb.bollinger_lband()
df['BB_Width'] = bb.bollinger_wband() # スクイーズ・エクスパンションの指標
df['BB_Width_Ratio'] = df['BB_Width'] / 100.0 # スケール調整
df['BB_Pos'] = bb.bollinger_pband() # バンド内の相対位置 (0~1)

# ヒゲの長さ
df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / df['Close']
df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / df['Close']

# 過去5日間のヒゲの平均 (上ヒゲをつけやすい/下ヒゲをつけやすい銘柄か？)
df['Upper_Shadow_5d_MA'] = df['Upper_Shadow_Ratio'].rolling(window=5).mean()
df['Lower_Shadow_5d_MA'] = df['Lower_Shadow_Ratio'].rolling(window=5).mean()

# 既存の基本的なオシレーター等
df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
macd = MACD(close=df['Close'])
df['MACD_Ratio'] = macd.macd() / df['Close']
df['MACD_Hist_Ratio'] = macd.macd_diff() / df['Close']
df['Vol_Change'] = df['Volume'].pct_change()

# 3. 予測ターゲットの設定 (高値と安値に特化)
# ターゲット: 当日の始値を基準とした、High/Lowの乖離率を予測する
# これにより、「終値は変わらなくても、日中どれくらい暴れるか」を直接学習させる
for i in range(1, 6):
    # i日後の 始値〜高値 の上昇幅 (%)
    df[f'Target_High_{i}d'] = (df['High'].shift(-i) - df['Open'].shift(-i)) / df['Open'].shift(-i)
    # i日後の 始値〜安値 の下落幅 (%) -> 安値はマイナスになる
    df[f'Target_Low_{i}d']  = (df['Low'].shift(-i) - df['Open'].shift(-i)) / df['Open'].shift(-i)
    
    # ちなみに、Open自体の予測は「前日CloseからのGap」として学習する（一応基準点が必要なため）
    df[f'Target_Open_Gap_{i}d'] = (df['Open'].shift(-i) - df['Close'].shift(-i+1)) / df['Close'].shift(-i+1)

base_features = [
    'Return', 'Vol_Change', 'RSI', 'MACD_Ratio', 'MACD_Hist_Ratio',
    'ATR_Ratio', 'BB_Width_Ratio', 'BB_Pos',
    'Upper_Shadow_Ratio', 'Lower_Shadow_Ratio',
    'Upper_Shadow_5d_MA', 'Lower_Shadow_5d_MA',
    'DJI_Return', 'N225_Return', 'USD_JPY_Return'
]

df = df.replace([np.inf, -np.inf], np.nan)

print("Training specialized Quantile Regression models for Extremes...")
# Quantile Regression の設定 (Highは90%, Lowは10%の分位点を狙う)
# 注意: XGBoostのQuantile損失関数は `reg:quantileerror` を使用します。

df_train = df.dropna(subset=base_features + [f'Target_High_{i}d' for i in range(1,6)] + [f'Target_Low_{i}d' for i in range(1,6)])
X_train = df_train[base_features]

models = {}
for i in range(1, 6):
    # --- 1. 高値 (High) 用モデル: 楽観的シナリオ (上振れ極値) ---
    y_high = df_train[f'Target_High_{i}d']
    model_high = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=0.90, # 90%分位点 (上ヒゲの先端付近を狙う)
        n_estimators=150, max_depth=3, learning_rate=0.05, 
        random_state=42, tree_method='hist'
    )
    model_high.fit(X_train, y_high)
    models[f'High_{i}d'] = model_high
    
    # --- 2. 安値 (Low) 用モデル: 悲観的シナリオ (下振れ極値) ---
    y_low = df_train[f'Target_Low_{i}d']
    model_low = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=0.10, # 10%分位点 (下ヒゲの先端付近を狙う)
        n_estimators=150, max_depth=3, learning_rate=0.05, 
        random_state=42, tree_method='hist'
    )
    model_low.fit(X_train, y_low)
    models[f'Low_{i}d'] = model_low
    
    # --- 3. 基準となる始値ギャップ (Open Gap) の予測 (平均的な回帰) ---
    y_open_gap = df_train[f'Target_Open_Gap_{i}d']
    model_open = xgb.XGBRegressor(
        objective='reg:squarederror', # Openは平均値を狙う
        n_estimators=100, max_depth=3, learning_rate=0.05, 
        random_state=42, tree_method='hist'
    )
    model_open.fit(X_train, y_open_gap)
    models[f'Open_Gap_{i}d'] = model_open

# 最新データでの推論
last_row = df.iloc[-1].copy()
last_row.fillna(0, inplace=True)
X_pred = pd.DataFrame([last_row[base_features]])

last_date = last_row['Date']
current_close = last_row['Close']

print("\nGenerating 5-day Micro Extremes predictions...")
predictions = []

# シミュレーションのための基準値トラッキング
prev_close = current_close

# 日付リストの作成 (土日スキップ)
next_days = []
tmp_date = last_date
while len(next_days) < 5:
    tmp_date += timedelta(days=1)
    if tmp_date.weekday() < 5:
        next_days.append(tmp_date)

for idx, target_date in enumerate(next_days):
    i = idx + 1
    
    # 1. まず「前日終値」から「当日始値」のGapを予測
    pred_open_gap = models[f'Open_Gap_{i}d'].predict(X_pred)[0]
    pred_open = prev_close * (1 + pred_open_gap)
    
    # 2. その始値を基準に、High(上振れ限界)とLow(下振れ限界)を予測
    pred_high_ratio = models[f'High_{i}d'].predict(X_pred)[0]
    pred_low_ratio = models[f'Low_{i}d'].predict(X_pred)[0]
    
    # High は Open より上、Low は Open より下であることを強制
    pred_high_ratio = max(pred_high_ratio, 0.0)
    pred_low_ratio = min(pred_low_ratio, 0.0)
    
    pred_high = pred_open * (1 + pred_high_ratio)
    pred_low = pred_open * (1 + pred_low_ratio)
    
    # (参考) Closeの予測は今回は重視しないが、一応 Open~High/Low の中間に仮置き
    # ※ または前日比の単純なモメンタムを置くが、極値予測特化モデルなのでCloseは省略気味に扱う
    pred_close = (pred_open + pred_high + pred_low) / 3.0 
    
    predictions.append({
        'Date': target_date.strftime('%Y-%m-%d'),
        'Day': f'{i}日後',
        'Pred_Low': np.round(pred_low, 0),
        'Pred_Open': np.round(pred_open, 0),
        'Pred_High': np.round(pred_high, 0),
        'Max_Volatility': f"{np.round((pred_high - pred_low) / pred_open * 100, 1)}%"
    })
    
    # 次の日のGap計算のために、今回のCloseを前日終値としてセット
    prev_close = pred_close

df_preds = pd.DataFrame(predictions)

print(f"\n=== ミクロ極値予測モデル (分位点回帰): {ticker} ===")
print(f"基準日: {last_date.strftime('%Y-%m-%d')}, 最新終値: {current_close:.0f}円")
print("※ 始値(Open)を基準とした、その日の最大上振れ(High:90%点)と最大下振れ(Low:10%点)に特化")
print("-" * 75)
print(df_preds[['Date', 'Day', 'Pred_Low', 'Pred_Open', 'Pred_High', 'Max_Volatility']].to_string(index=False))

# グラフ描画
fig = go.Figure()

df_plot = df.tail(20)
fig.add_trace(go.Candlestick(
    x=df_plot['Date'],
    open=df_plot['Open'], high=df_plot['High'],
    low=df_plot['Low'], close=df_plot['Close'],
    name='Actual (過去20日)',
    increasing_line_color='black', decreasing_line_color='black',
    increasing_fillcolor='white', decreasing_fillcolor='black'
))

# 予測範囲のエラーバー的な表現
df_preds['Date'] = pd.to_datetime(df_preds['Date'])
fig.add_trace(go.Candlestick(
    x=df_preds['Date'],
    open=df_preds['Pred_Open'], high=df_preds['Pred_High'],
    low=df_preds['Pred_Low'], close=df_preds['Pred_Open'], # Closeは便宜上Openと同じにして「ヒゲ」を強調
    name='Predicted Extremes (予測極値)',
    increasing_line_color='red', decreasing_line_color='blue',
    increasing_fillcolor='rgba(255, 0, 0, 0.2)', decreasing_fillcolor='rgba(0, 0, 255, 0.2)'
))

fig.update_layout(
    title=f'【極値特化モデル】 {ticker} 5日間の高値・安値 予測帯',
    yaxis_title='株価 (円)',
    xaxis_rangeslider_visible=False,
    height=600, width=1000, template='plotly_white'
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_html = f"predict_extremes_{ticker.replace('.T','')}_{timestamp}.html"
fig.write_html(output_html)
print(f"\n=> 予測チャートを {output_html} に保存しました。")

