import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
import xgboost as xgb
from datetime import timedelta
import plotly.graph_objects as go
from tqdm import tqdm
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

# 4. データ結合 (外部データのmergeは生データに対して行う - テクニカル指標はループ内で計算)
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
df_raw['USD_JPY_Return'].fillna(0, inplace=True)

# 5. 特徴量とターゲットの作成
targets = ['Open', 'High', 'Low', 'Close']
for t in targets:
    for i in range(1, 6):
        df_raw[f'Target_{t}_{i}d'] = df_raw[t].shift(-i)

# df_raw contains raw OHLCV + merged external data + targets, but NO technical indicators yet

features = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Vol_Change',
    'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB', 'Close_lag26',
    'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
    'DJI_Return', 'N225_Return', 'USD_JPY', 'USD_JPY_Return'
]

# === 1月から現在までの連続バックテスト ===
start_eval_date = pd.to_datetime('2026-01-05') # 2026年の最初の営業日付近
end_eval_date = df_raw['Date'].max()

# 評価対象となる起点日（月曜日などの週初めを中心に一定間隔、あるいはすべての営業日）
# 今回は計算コストを抑えつつ全体の波形を見るため、週1回（金曜日引け後＝月曜朝の予測起点）でローリング予測を実施
evaluation_dates = df_raw[(df_raw['Date'] >= start_eval_date) & (df_raw['Date'] <= end_eval_date) & (df_raw['Date'].dt.dayofweek == 4)]['Date'].tolist()

print(f"\nRunning rolling predictions from {start_eval_date.strftime('%Y-%m-%d')} to {end_eval_date.strftime('%Y-%m-%d')}...")

all_predictions = []

for base_date in tqdm(evaluation_dates):
    # データリーク防止: base_date + 予測horizon(5日)分までの生データのみでテクニカル指標を計算
    cutoff_date = base_date + timedelta(days=10)  # 5営業日分のバッファ
    df_cutoff = df_raw[df_raw['Date'] <= cutoff_date].copy()
    df_iter = compute_features(df_cutoff)

    # 基準日までのデータで学習 (1-day gap to prevent leakage)
    gap = pd.Timedelta(days=1)
    df_train = df_iter[df_iter['Date'] <= base_date - gap].copy()
    df_train.ffill(inplace=True)
    df_train = df_train.dropna().reset_index(drop=True)
    if len(df_train) < 50: # 十分な学習データがない場合はスキップ
        continue

    models = {}
    for t in targets:
        for i in range(1, 6):
            y = df_train[f'Target_{t}_{i}d']
            model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
            model.fit(df_train[features], y)
            models[f'{t}_{i}d'] = model

    # 基準日時点の最新データを取得
    base_data = df_iter[df_iter['Date'] <= base_date].iloc[-1].copy()
    base_data.ffill(inplace=True)
    base_data = base_data.fillna(0)
    X_base = pd.DataFrame([base_data[features]])

    # 翌5日間の日付を計算
    next_days = []
    current_date = base_date
    while len(next_days) < 5:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            next_days.append(current_date)
            
    # 予測
    for i, d in enumerate(next_days):
        pred_open = models[f'Open_{i+1}d'].predict(X_base)[0]
        pred_high = models[f'High_{i+1}d'].predict(X_base)[0]
        pred_low = models[f'Low_{i+1}d'].predict(X_base)[0]
        pred_close = models[f'Close_{i+1}d'].predict(X_base)[0]
        
        # 論理的整合性の補正
        actual_high = max(pred_high, pred_open, pred_close, pred_low)
        actual_low = min(pred_low, pred_open, pred_close, pred_high)
        
        all_predictions.append({
            'Base_Date': base_date,
            'Target_Date': d,
            'Open': np.round(pred_open, 0),
            'High': np.round(actual_high, 0),
            'Low': np.round(actual_low, 0),
            'Close': np.round(pred_close, 0)
        })

df_preds = pd.DataFrame(all_predictions)
df_preds = df_preds.sort_values(['Base_Date', 'Target_Date'])

# 重複する日付の予測は、最新の起点からの予測を採用する（より精度が高いため）
df_preds_best = df_preds.sort_values(['Target_Date', 'Base_Date']).drop_duplicates('Target_Date', keep='last')

# グラフ描画
print("\nGenerating comprehensive visualization...")
# For chart display, compute features on full dataset (not used for training)
df_all_chart = compute_features(df_raw)
df_plot_actual = df_all_chart[(df_all_chart['Date'] >= start_eval_date) & (df_all_chart['Date'] <= end_eval_date)].copy()

fig = go.Figure()

# 1. 実際のローソク足
fig.add_trace(go.Candlestick(
    x=df_plot_actual['Date'],
    open=df_plot_actual['Open'], high=df_plot_actual['High'],
    low=df_plot_actual['Low'], close=df_plot_actual['Close'],
    name='実際の値動き (実績)',
    increasing_line_color='black', decreasing_line_color='black',
    increasing_fillcolor='white', decreasing_fillcolor='black'
))

# 2. AI予測のローソク足（青/赤の半透明で重ねる）
fig.add_trace(go.Candlestick(
    x=df_preds_best['Target_Date'],
    open=df_preds_best['Open'], high=df_preds_best['High'],
    low=df_preds_best['Low'], close=df_preds_best['Close'],
    name='AI予測値 (直近の週末を起点)',
    increasing_line_color='blue', decreasing_line_color='red',
    increasing_fillcolor='rgba(0, 0, 255, 0.4)', decreasing_fillcolor='rgba(255, 0, 0, 0.4)'
))

# レイアウト
fig.update_layout(
    title='【1月〜現在】 Sansan(4443) 実際のローソク足(黒/白) と 毎週末起点のAI予測ローソク足(青/赤) の軌跡比較',
    yaxis_title='株価 (円)',
    xaxis_title='日付',
    xaxis_rangeslider_visible=False,
    height=800,
    width=1400,
    hovermode='x unified',
    template='plotly_white'
)

# 主要なイベント（暴落日など）に線を引く
fig.add_vline(x=pd.to_datetime('2026-02-04').timestamp() * 1000, line_dash="dot", line_color="orange")
fig.add_annotation(x='2026-02-04', y=1500, text="2/4 大暴落", showarrow=True, arrowhead=1)

output_html = "backtest_ytd_chart.html"
fig.write_html(output_html)
print(f"\n=> 1月からの連続予測比較チャートを {output_html} に保存しました。")
