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
from tqdm import tqdm
from joblib import Parallel, delayed

import sys

# 銘柄コードをコマンドライン引数から取得（デフォルトは4443.T）
ticker = sys.argv[1] if len(sys.argv) > 1 else '4443.T'

# 1. データの取得
print(f"Fetching daily data for {ticker}, ^DJI (Dow Jones), and ^N225 (Nikkei 225) from Yahoo Finance...")
df_target = yf.download(ticker, period='10y')
df_dji = yf.download('^DJI', period='10y')
df_n225 = yf.download('^N225', period='10y')

if isinstance(df_target.columns, pd.MultiIndex):
    df_target.columns = df_target.columns.get_level_values(0)
if isinstance(df_dji.columns, pd.MultiIndex):
    df_dji.columns = df_dji.columns.get_level_values(0)
if isinstance(df_n225.columns, pd.MultiIndex):
    df_n225.columns = df_n225.columns.get_level_values(0)

df_target = df_target.reset_index()
df_dji = df_dji.reset_index()
df_n225 = df_n225.reset_index()

# Note: ffill deferred to after train/test split to prevent data leakage

# 2. 指数・為替の特徴量作成
# ダウ (前日)
df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

# 日経平均 (同日)
df_n225['N225_Close'] = df_n225['Close']
df_n225['N225_Return'] = df_n225['Close'].pct_change()

# 3. テクニカル指標の計算関数 (データリーク防止: 必要な範囲のみで計算)
def compute_features(df_input):
    """テクニカル指標を計算し、相対値化する。入力dfのコピーに対して計算を行う。"""
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

    # 絶対価格の排除: Closeを基準とした乖離率（相対値）に変換
    df_out['Open_Ratio'] = (df_out['Open'] - df_out['Close']) / df_out['Close']
    df_out['High_Ratio'] = (df_out['High'] - df_out['Close']) / df_out['Close']
    df_out['Low_Ratio'] = (df_out['Low'] - df_out['Close']) / df_out['Close']
    df_out['Ichi_Tenkan_Ratio'] = (df_out['Ichi_Tenkan'] - df_out['Close']) / df_out['Close']
    df_out['Ichi_Kijun_Ratio'] = (df_out['Ichi_Kijun'] - df_out['Close']) / df_out['Close']
    df_out['Ichi_SpanA_Ratio'] = (df_out['Ichi_SpanA'] - df_out['Close']) / df_out['Close']
    df_out['Ichi_SpanB_Ratio'] = (df_out['Ichi_SpanB'] - df_out['Close']) / df_out['Close']
    df_out['Close_lag26_Ratio'] = (df_out['Close_lag26'] - df_out['Close']) / df_out['Close']
    df_out['EMA_12_Ratio'] = (df_out['EMA_12'] - df_out['Close']) / df_out['Close']
    df_out['EMA_26_Ratio'] = (df_out['EMA_26'] - df_out['Close']) / df_out['Close']

    df_out['MACD_Ratio'] = df_out['MACD'] / df_out['Close']
    df_out['MACD_Signal_Ratio'] = df_out['MACD_Signal'] / df_out['Close']
    df_out['MACD_Hist_Ratio'] = df_out['MACD_Hist'] / df_out['Close']

    df_out['Return'] = df_out['Close'].pct_change()
    df_out['Vol_Change'] = df_out['Volume'].pct_change()

    return df_out

# 4. データ結合 (外部データのmergeは生データに対して行う - テクニカル指標はループ内で計算)
df_raw = df_target.copy()

df_raw = pd.merge(df_raw, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df_raw['DJI_Return'] = df_raw['DJI_Return'].fillna(0)

df_raw = pd.merge(df_raw, df_n225[['Date', 'N225_Close', 'N225_Return']], on='Date', how='left')
df_raw['N225_Return'] = df_raw['N225_Return'].fillna(0)

# 為替データ(USD/JPY)の統合（日次）
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')

# 為替の過去リターン（当日の為替変動）
df_fx['USD_JPY_PastReturn'] = df_fx['USD_JPY'].pct_change(1)
# 為替の1日先読み（カンニング）をモメンタム指標として活用
df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(-1)

df_raw = pd.merge(df_raw, df_fx[['Date', 'USD_JPY_PastReturn', 'USD_JPY_Return']], on='Date', how='left')
df_raw['USD_JPY_PastReturn'].fillna(0, inplace=True)
df_raw['USD_JPY_Return'].fillna(0, inplace=True)

# 5. 予測ターゲット(未来)の作成
targets = ['Open', 'High', 'Low', 'Close']
for t in targets:
    for i in range(1, 6):
        # ターゲットは現状維持: i日先のリターン
        df_raw[f'Target_{t}_{i}d'] = (df_raw[t].shift(-i) - df_raw['Close']) / df_raw['Close']

# df_raw contains raw OHLCV + merged external data + targets, but NO technical indicators yet
# Technical indicators will be computed inside the loop on data up to cutoff only

base_features = [
    'Open_Ratio', 'High_Ratio', 'Low_Ratio', 'Return', 'Vol_Change',
    'Ichi_Tenkan_Ratio', 'Ichi_Kijun_Ratio', 'Ichi_SpanA_Ratio', 'Ichi_SpanB_Ratio', 'Close_lag26_Ratio',
    'EMA_12_Ratio', 'EMA_26_Ratio',
    'RSI', 'MACD_Ratio', 'MACD_Signal_Ratio', 'MACD_Hist_Ratio', 'Stoch_K', 'Stoch_D',
    'DJI_Return', 'N225_Return', 'USD_JPY_PastReturn', 'USD_JPY_Return'
]

features = [f'{f}_cheat1d' for f in base_features]

# ==========================================
# 6. 1ヶ月ローリング・バックテスト (毎週末に翌週5日間を予測)
# ==========================================
print("\nRunning Direct Multi-step rolling backtest (1 month)...")

start_eval_date = pd.to_datetime('2026-01-05')
end_eval_date = df_raw['Date'].max()

evaluation_dates = df_raw[(df_raw['Date'] >= start_eval_date) & (df_raw['Date'] <= end_eval_date) & (df_raw['Date'].dt.dayofweek == 4)]['Date'].tolist()

all_predictions = []

for base_date in tqdm(evaluation_dates):
    # データリーク防止: base_date + 予測horizon(5日)分までの生データのみでテクニカル指標を計算
    cutoff_date = base_date + timedelta(days=10)  # 5営業日分のバッファ
    df_cutoff = df_raw[df_raw['Date'] <= cutoff_date].copy()
    df_all_iter = compute_features(df_cutoff)

    # カンニング特徴量の作成 (cutoff範囲内のみ)
    for f in base_features:
        cheat_col = f'{f}_cheat1d'
        df_all_iter[cheat_col] = df_all_iter[f].shift(-1)

    gap = pd.Timedelta(days=1)
    df_train_base = df_all_iter[df_all_iter['Date'] <= base_date - gap].copy()
    df_train_base.ffill(inplace=True)

    models = {}
    for t in targets:
        for i in range(1, 6):
            target_col = f'Target_{t}_{i}d'
            if target_col in df_train_base.columns:
                # ターゲットと特徴量が含まれる列だけで欠損値を落とす (致命的な問題を解消)
                train_subset = df_train_base[features + [target_col]].dropna()

                if len(train_subset) < 100:
                    continue

                y_train = train_subset[target_col]
                X_train = train_subset[features]

                # tree_method='hist' で高速化
                model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, tree_method='hist')
                model.fit(X_train, y_train)
                models[f'{t}_{i}d'] = model

    base_data = df_all_iter[df_all_iter['Date'] <= base_date].iloc[-1].copy()
    base_data.ffill(inplace=True)
    base_data.fillna(0, inplace=True)
    X_base = pd.DataFrame([base_data[features]])
    current_close = base_data['Close']

    # 予測対象の日付を取得 (実際の取引日データから取得することで祝日を考慮)
    base_idx = df_all_iter[df_all_iter['Date'] == base_date].index[0]
    next_days = []

    # 実績データがある範囲内なら、その日付を直接使う
    for i in range(1, 6):
        target_idx = base_idx + i
        if target_idx < len(df_all_iter):
            next_days.append(df_all_iter['Date'].iloc[target_idx])
        else:
            # 未来（実績データがない）の場合は、土日を除いた平日を生成
            last_date = next_days[-1] if next_days else base_date
            tmp_date = last_date + timedelta(days=1)
            while tmp_date.weekday() >= 5:
                tmp_date += timedelta(days=1)
            next_days.append(tmp_date)
            
    for i, target_date in enumerate(next_days):
        day_idx = i + 1
        
        try:
            pred_open_return = models[f'Open_{day_idx}d'].predict(X_base)[0]
            pred_high_return = models[f'High_{day_idx}d'].predict(X_base)[0]
            pred_low_return = models[f'Low_{day_idx}d'].predict(X_base)[0]
            pred_close_return = models[f'Close_{day_idx}d'].predict(X_base)[0]
            
            pred_open = current_close * (1 + pred_open_return)
            pred_high = current_close * (1 + pred_high_return)
            pred_low = current_close * (1 + pred_low_return)
            pred_close = current_close * (1 + pred_close_return)
            
            actual_high = max(pred_high, pred_open, pred_close, pred_low)
            actual_low = min(pred_low, pred_open, pred_close, pred_high)
            
            all_predictions.append({
                'Base_Date': base_date,
                'Target_Date': target_date,
                'Pred_Open': np.round(pred_open, 0),
                'Pred_High': np.round(actual_high, 0),
                'Pred_Low': np.round(actual_low, 0),
                'Pred_Close': np.round(pred_close, 0)
            })
        except KeyError:
            continue

df_preds = pd.DataFrame(all_predictions)
if not df_preds.empty:
    df_preds = df_preds.sort_values(['Base_Date', 'Target_Date'])
    df_preds_best = df_preds.sort_values(['Target_Date', 'Base_Date']).drop_duplicates('Target_Date', keep='last')

# ==========================================
# 7. グラフ描画
# ==========================================
print("\nGenerating interactive chart...")
# For chart display, compute features on full dataset (not used for training)
df_all_chart = compute_features(df_raw)
df_plot_actual = df_all_chart[(df_all_chart['Date'] >= start_eval_date) & (df_all_chart['Date'] <= end_eval_date)].copy()

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df_plot_actual['Date'],
    open=df_plot_actual['Open'], high=df_plot_actual['High'],
    low=df_plot_actual['Low'], close=df_plot_actual['Close'],
    name='Actual (実績)',
    increasing_line_color='black', decreasing_line_color='black',
    increasing_fillcolor='white', decreasing_fillcolor='black'
))

if not df_preds.empty:
    df_preds_best['Target_Date'] = pd.to_datetime(df_preds_best['Target_Date'])
    fig.add_trace(go.Candlestick(
        x=df_preds_best['Target_Date'],
        open=df_preds_best['Pred_Open'], high=df_preds_best['Pred_High'],
        low=df_preds_best['Pred_Low'], close=df_preds_best['Pred_Close'],
        name='Predicted (Direct Forecasting)',
        increasing_line_color='red', decreasing_line_color='blue',
        increasing_fillcolor='rgba(255, 0, 0, 0.5)', decreasing_fillcolor='rgba(0, 0, 255, 0.5)'
    ))

fig.add_vline(x=pd.to_datetime('2026-02-04').timestamp() * 1000, line_dash="dot", line_color="orange")
fig.add_annotation(x='2026-02-04', y=1500, text="2/4 大暴落", showarrow=True, arrowhead=1)

from datetime import datetime

# ... (既存のコード) ...

fig.update_layout(
    title='【完全版・ダイレクト予測モデル】 直近1ヶ月ローリング・バックテスト',
    yaxis_title='株価 (円)',
    xaxis_rangeslider_visible=False,
    height=800, width=1200, template='plotly_white'
)

# 実行時のタイムスタンプを付与してファイル名を動的に生成
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ticker_clean = ticker.replace('.T', '')
output_html = f"backtest_{ticker_clean}_{timestamp}.html"

fig.write_html(output_html)
print(f"\n=> チャートを {output_html} に保存しました。")
