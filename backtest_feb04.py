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

# 4. データ結合 (外部データのmergeは生データに対して行う - テクニカル指標は分割後に計算)
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
targets = ['Open', 'High', 'Low', 'Close']
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

# === 暴落した日 (2/4) を起点に翌日以降の5日間を予測 ===
base_date_str = '2026-02-04'
test_date_start = pd.to_datetime(base_date_str) + pd.Timedelta(days=1)

print(f"\nTraining models for OHLC prediction starting after {base_date_str}...")

# データリーク防止: 基準日 + 予測horizon分までの生データのみでテクニカル指標を計算
cutoff_date = pd.to_datetime(base_date_str) + timedelta(days=10)  # 5営業日分のバッファ
df_cutoff = df_raw[df_raw['Date'] <= cutoff_date].copy()
df_all = compute_features(df_cutoff)

# 基準日までのデータで学習 (1-day gap to prevent leakage)
gap = pd.Timedelta(days=1)
df_train = df_all[df_all['Date'] <= pd.to_datetime(base_date_str) - gap].copy()
df_train.ffill(inplace=True)
df_train = df_train.dropna().reset_index(drop=True)

models = {}
for t in targets:
    for i in range(1, 6):
        y = df_train[f'Target_{t}_{i}d']
        model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(df_train[features], y)
        models[f'{t}_{i}d'] = model

# 基準日時点のデータを取得
base_data = df_all[df_all['Date'] <= pd.to_datetime(base_date_str)].iloc[-1].copy()
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
predictions_ohlc = {
    'Date': [],
    'Open': [],
    'High': [],
    'Low': [],
    'Close': [],
    'Direction_Pred': []
}

prev_close_pred = base_data['Close']

for d in next_days:
    predictions_ohlc['Date'].append(d.strftime('%Y-%m-%d'))

for i in range(1, 6):
    pred_open = models[f'Open_{i}d'].predict(X_base)[0]
    pred_high = models[f'High_{i}d'].predict(X_base)[0]
    pred_low = models[f'Low_{i}d'].predict(X_base)[0]
    pred_close = models[f'Close_{i}d'].predict(X_base)[0]
    
    actual_high = max(pred_high, pred_open, pred_close, pred_low)
    actual_low = min(pred_low, pred_open, pred_close, pred_high)
    
    predictions_ohlc['Open'].append(np.round(pred_open, 0))
    predictions_ohlc['High'].append(np.round(actual_high, 0))
    predictions_ohlc['Low'].append(np.round(actual_low, 0))
    predictions_ohlc['Close'].append(np.round(pred_close, 0))
    
    direction = "UP" if pred_close > prev_close_pred else "DOWN"
    predictions_ohlc['Direction_Pred'].append(direction)
    prev_close_pred = pred_close

pred_df = pd.DataFrame(predictions_ohlc)

print(f"\n=== 起点: {base_date_str} (大暴落日 / 終値: {base_data['Close']}円) からのOHLC予測と検証 ===")
print("-------------------------------------------------")
print(pred_df[['Date', 'Open', 'High', 'Low', 'Close', 'Direction_Pred']].to_string(index=False))

correct_in_batch = 0
valid_days = 0
prev_actual_close = base_data['Close']

print("\n--- 実績との比較 ---")
for i, d in enumerate(next_days):
    d_str = d.strftime('%Y-%m-%d')
    actual = df_all[df_all['Date'] == d_str]
    
    if not actual.empty:
        a_open = actual['Open'].values[0]
        a_high = actual['High'].values[0]
        a_low = actual['Low'].values[0]
        a_close = actual['Close'].values[0]
        
        actual_dir = "UP" if a_close > prev_actual_close else "DOWN"
        pred_dir = pred_df['Direction_Pred'].iloc[i]
        
        is_correct = "O" if actual_dir == pred_dir else "X"
        if is_correct == "O":
            correct_in_batch += 1
        valid_days += 1
        
        p = predictions_ohlc
        print(f"\n[{d_str}] 方向性予測: {pred_dir} | 実績: {actual_dir} -> 判定: {is_correct}")
        print(f"       [ 始値(Open) | 高値(High) | 安値(Low) | 終値(Close) ]")
        print(f"  予測: {p['Open'][i]:>10.0f} | {p['High'][i]:>10.0f} | {p['Low'][i]:>9.0f} | {p['Close'][i]:>11.0f}")
        print(f"  実績: {a_open:>10.0f} | {a_high:>10.0f} | {a_low:>9.0f} | {a_close:>11.0f}")
        print(f"  誤差: {abs(p['Open'][i]-a_open):>10.0f} | {abs(p['High'][i]-a_high):>10.0f} | {abs(p['Low'][i]-a_low):>9.0f} | {abs(p['Close'][i]-a_close):>11.0f}")
        
        prev_actual_close = a_close

if valid_days > 0:
    print(f"\n-> 方向性の正答率: {correct_in_batch}/{valid_days} ({correct_in_batch/valid_days*100:.1f}%)")

import plotly.graph_objects as go

# 実際の過去データ（直近30日分程度）
df_plot = df_all[df_all['Date'] <= '2026-02-12'].tail(30).copy()

fig = go.Figure()

# 1. 実際のローソク足（過去から2/12まで）
fig.add_trace(go.Candlestick(
    x=df_plot['Date'],
    open=df_plot['Open'], high=df_plot['High'],
    low=df_plot['Low'], close=df_plot['Close'],
    name='実際の値動き (実績)',
    increasing_line_color='black', decreasing_line_color='black',
    increasing_fillcolor='white', decreasing_fillcolor='black'
))

# 2. AI予測のローソク足（2/5 〜 2/11）
# 予測データをDFに変換
pred_dates = pd.to_datetime(pred_df['Date'])
fig.add_trace(go.Candlestick(
    x=pred_dates,
    open=pred_df['Open'], high=pred_df['High'],
    low=pred_df['Low'], close=pred_df['Close'],
    name='AI予測値 (2/4起点)',
    increasing_line_color='red', decreasing_line_color='blue',
    increasing_fillcolor='rgba(255, 0, 0, 0.5)', decreasing_fillcolor='rgba(0, 0, 255, 0.5)'
))

# グラフのレイアウト調整
fig.update_layout(
    title='【バックテスト検証】 2/4大暴落日を起点とした四本値(OHLC)のAI予測と実績の比較',
    yaxis_title='株価 (円)',
    xaxis_title='日付',
    xaxis_rangeslider_visible=False,
    height=800,
    width=1200,
    hovermode='x unified',
    template='plotly_white'
)

# 2/4の暴落日に縦線を引く
# plotlyのadd_vlineで文字列日付を扱う際のエラー回避のため、pd.to_datetimeを使用するか、注釈のテキスト設定を変更する
fig.add_vline(x=pd.to_datetime('2026-02-04').timestamp() * 1000, line_dash="dash", line_color="green")
fig.add_annotation(x='2026-02-04', y=1350, text="予測起点 (2/4)", showarrow=True, arrowhead=1)

output_html = "backtest_feb04_chart.html"
fig.write_html(output_html)
print(f"\n=> 予測と実績の比較チャートを {output_html} に保存しました。")
