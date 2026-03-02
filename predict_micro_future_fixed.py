import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. データの取得と前処理
# ==========================================
print("Fetching daily data for 4443.T (Sansan), ^DJI (Dow Jones), and ^N225 (Nikkei 225) from Yahoo Finance...")
df_sansan = yf.download('4443.T', period='2y')
df_dji = yf.download('^DJI', period='2y')
df_n225 = yf.download('^N225', period='2y')

for df_temp in [df_sansan, df_dji, df_n225]:
    if isinstance(df_temp.columns, pd.MultiIndex):
        df_temp.columns = df_temp.columns.get_level_values(0)

df_sansan = df_sansan.reset_index()
df_dji = df_dji.reset_index()
df_n225 = df_n225.reset_index()

df_sansan.ffill(inplace=True)
df_dji.ffill(inplace=True)
df_n225.ffill(inplace=True)

# 2. 指数・為替の特徴量作成 (定常性を重視し、リターンに変換)
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

df_n225['N225_Return'] = df_n225['Close'].pct_change()

# 3. テクニカル指標の計算 (Sansan)
df = df_sansan.copy()

# 変化率とボラティリティ
df['Return'] = df['Close'].pct_change()
df['Vol_Change'] = df['Volume'].pct_change()
df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Low']
df['Open_Close_Spread'] = (df['Close'] - df['Open']) / df['Open']

# テクニカル指標 (そのまま使用可能な定常・オシレータ系)
df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
df['Stoch_K'] = stoch.stoch()
df['Stoch_D'] = stoch.stoch_signal()

# 非定常指標（MACD, 一目均衡表, EMA）は「価格からの乖離率(%)」に変換して定常化する
macd = MACD(close=df['Close'])
df['MACD_Diff_Pct'] = macd.macd_diff() / df['Close'] * 100

ichi = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
df['Ichi_Tenkan_Div'] = (df['Close'] - ichi.ichimoku_conversion_line()) / ichi.ichimoku_conversion_line() * 100
df['Ichi_Kijun_Div'] = (df['Close'] - ichi.ichimoku_base_line()) / ichi.ichimoku_base_line() * 100

df['EMA_12_Div'] = (df['Close'] - df['Close'].ewm(span=12, adjust=False).mean()) / df['Close'].ewm(span=12, adjust=False).mean() * 100

# 過去の生リターンのラグ（自己回帰）
for lag in [1, 2, 3]:
    df[f'Return_lag_{lag}'] = df['Return'].shift(lag)

# 為替データ (先読みバイアスの修正)
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx = df_fx.sort_values('Date') # 昇順ソート
df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(1) # 修正: pct_change(1)に変更

# 4. データ結合とNaN処理
df = pd.merge(df, df_dji[['Date_JP', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df = pd.merge(df, df_n225[['Date', 'N225_Return']], on='Date', how='left')
df = pd.merge(df, df_fx[['Date', 'USD_JPY_Return']], on='Date', how='left')

df.ffill(inplace=True)
df.fillna(0, inplace=True)

# ==========================================
# 5. 予測ターゲットの作成 (リターンを予測)
# ==========================================
df['Target_Open_Return'] = (df['Open'].shift(-1) - df['Close']) / df['Close']
df['Target_High_Return'] = (df['High'].shift(-1) - df['Close']) / df['Close']
df['Target_Low_Return']  = (df['Low'].shift(-1) - df['Close']) / df['Close']
df['Target_Close_Return']= (df['Close'].shift(-1) - df['Close']) / df['Close']

targets = ['Target_Open_Return', 'Target_High_Return', 'Target_Low_Return', 'Target_Close_Return']

features = [
    'Return', 'Vol_Change', 'High_Low_Spread', 'Open_Close_Spread',
    'RSI', 'Stoch_K', 'Stoch_D', 'MACD_Diff_Pct',
    'Ichi_Tenkan_Div', 'Ichi_Kijun_Div', 'EMA_12_Div',
    'Return_lag_1', 'Return_lag_2', 'Return_lag_3',
    'DJI_Return', 'N225_Return', 'USD_JPY_Return'
]

df_all = df.copy()
test_date_start = pd.to_datetime('2026-03-03') # 本日3/2までのデータを学習に使う
df_train = df_all[df_all['Date'] < test_date_start].dropna(subset=targets).reset_index(drop=True)

# ==========================================
# 6. XGBoostモデルの評価 (TimeSeriesSplit)
# ==========================================
print("\n=== 時系列K-Fold CVによるXGBoost(単体)のOHLC独立評価 ===")
tscv = TimeSeriesSplit(n_splits=3)

for i in range(1, 6):
    print(f"\n--- {i}日先 (Target {i}d) の予測精度 ---")
    for t in targets:
        target_col = f'{t}_{i}d'
        base_col = t.replace('Target_', '').replace('_Return', '')
        
        # 評価用の一時的なターゲットを作成
        df_cv = df_train.copy()
        df_cv[target_col] = (df_cv[base_col].shift(-i) - df_cv['Close']) / df_cv['Close']
        df_cv = df_cv.dropna(subset=[target_col])
        
        X_cv = df_cv[features]
        y_cv = df_cv[target_col]
        
        rmses = []
        for train_idx, val_idx in tscv.split(X_cv):
            X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
            y_tr, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]
            
            model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmses.append(rmse)
        print(f"[{t}] CV RMSE: {np.mean(rmses):.4f}")

# ==========================================
# 7. 全データで最終学習
# ==========================================
print("\nTraining final XGBoost models...")
trained_models = {}
for t in targets:
    trained_models[t] = {}
    for i in range(1, 6): # 1dから5dまでのターゲットを作成して学習
        target_col = f'{t}_{i}d'
        # i日先の終値リターン等を計算して一時的に追加
        base_col = t.replace('Target_', '').replace('_Return', '')
        df_train[target_col] = (df_train[base_col].shift(-i) - df_train['Close']) / df_train['Close']
        
        # NaNを落として学習
        train_subset = df_train.dropna(subset=[target_col])
        y_train = train_subset[target_col]
        X_train = train_subset[features]
        
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        trained_models[t][f'{i}d'] = model

# ==========================================
# 8. 再帰的予測 (動的特徴量更新)
# ==========================================
raw_closes = list(df_all[df_all['Date'] < test_date_start]['Close'].tail(15).values)
raw_highs = list(df_all[df_all['Date'] < test_date_start]['High'].tail(15).values)
raw_lows = list(df_all[df_all['Date'] < test_date_start]['Low'].tail(15).values)

base_data = df_all[df_all['Date'] < test_date_start].iloc[-1].copy()
base_date = base_data['Date']
current_close = base_data['Close']
current_features = {f: base_data[f] for f in features}

# 各種指標の初期化（ループ前）
hist_close = df_all[df_all['Date'] < test_date_start]['Close']
current_ema12 = hist_close.ewm(span=12, adjust=False).mean().iloc[-1]

diff = hist_close.diff()
up = diff.where(diff > 0, 0)
down = -diff.where(diff < 0, 0)
current_avg_gain = up.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
current_avg_loss = down.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

stoch_k_list = df_all[df_all['Date'] < test_date_start]['Stoch_K'].tail(2).tolist()

next_days = []
tmp_date = base_date
while len(next_days) < 5:
    tmp_date += timedelta(days=1)
    if tmp_date.weekday() < 5:
        next_days.append(tmp_date)

predictions_ohlc = {'Date': [], 'Open': [], 'High': [], 'Low': [], 'Close': []}

for i, d in enumerate(next_days):
    predictions_ohlc['Date'].append(d.strftime('%Y-%m-%d'))
    
    X_pred = pd.DataFrame([current_features])
    preds_return = {}
    
    # ターゲット(OHLC)ごとの予測
    for t in targets:
        pred_val = trained_models[t][f'{i+1}d'].predict(X_pred)[0]
        preds_return[t] = pred_val
        
    p_open = current_close * (1 + preds_return['Target_Open_Return'])
    p_high = current_close * (1 + preds_return['Target_High_Return'])
    p_low = current_close * (1 + preds_return['Target_Low_Return'])
    p_close = current_close * (1 + preds_return['Target_Close_Return'])
    
    actual_high = max(p_high, p_open, p_close, p_low)
    actual_low = min(p_low, p_open, p_close, p_high)
    
    predictions_ohlc['Open'].append(np.round(p_open, 0))
    predictions_ohlc['High'].append(np.round(actual_high, 0))
    predictions_ohlc['Low'].append(np.round(actual_low, 0))
    predictions_ohlc['Close'].append(np.round(p_close, 0))
    
    # 動的特徴量更新
    pred_return = preds_return['Target_Close_Return']
    current_features['Return_lag_3'] = current_features['Return_lag_2']
    current_features['Return_lag_2'] = current_features['Return_lag_1']
    current_features['Return_lag_1'] = current_features['Return']
    current_features['Return'] = pred_return
    
    current_features['High_Low_Spread'] = (actual_high - actual_low) / actual_low
    current_features['Open_Close_Spread'] = (p_close - p_open) / p_open
    
    # RSIの正確な更新 (Wilder平滑化)
    change = p_close - raw_closes[-1]
    gain = max(change, 0)
    loss = max(-change, 0)
    current_avg_gain = (current_avg_gain * 13 + gain) / 14
    current_avg_loss = (current_avg_loss * 13 + loss) / 14
    if current_avg_loss == 0:
        rsi = 100
    else:
        rs = current_avg_gain / current_avg_loss
        rsi = 100 - (100 / (1 + rs))
    current_features['RSI'] = rsi
    
    raw_closes.append(p_close); raw_closes.pop(0)
    
    raw_highs.append(actual_high); raw_highs.pop(0)
    raw_lows.append(actual_low); raw_lows.pop(0)
    
    recent_14_high = max(raw_highs[-14:])
    recent_14_low = min(raw_lows[-14:])
    if recent_14_high == recent_14_low:
        stoch_k = 50.0
    else:
        stoch_k = 100 * (p_close - recent_14_low) / (recent_14_high - recent_14_low)
    current_features['Stoch_K'] = stoch_k
    
    # Stochastic D は3日間のSMA
    stoch_k_list.append(stoch_k)
    stoch_k_list.pop(0)
    current_features['Stoch_D'] = np.mean(stoch_k_list)
    
    alpha = 2 / (12 + 1)
    new_ema12 = (p_close - current_ema12) * alpha + current_ema12
    current_features['EMA_12_Div'] = (p_close - new_ema12) / new_ema12 * 100
    current_ema12 = new_ema12
    
    current_features['DJI_Return'] = 0.0
    current_features['N225_Return'] = 0.0
    current_features['USD_JPY_Return'] = 0.0
    
    current_close = p_close

pred_df = pd.DataFrame(predictions_ohlc)

print(f"\n=== 【リーク排除・XGBoost単体モデル】 本日({base_date.strftime('%m/%d')})起点 ===")
print(f"基準日(最新終値): {base_date.strftime('%Y-%m-%d')} ({base_data['Close']} 円)")
print("-------------------------------------------------")
print(pred_df.to_string(index=False))

# グラフ描画
df_plot = df_all[df_all['Date'] <= base_date].tail(30).copy()
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df_plot['Date'], open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'],
    name='実際の値動き (実績)', increasing_line_color='black', decreasing_line_color='black'
))

pred_dates = pd.to_datetime(pred_df['Date'])
fig.add_trace(go.Candlestick(
    x=pred_dates, open=pred_df['Open'], high=pred_df['High'], low=pred_df['Low'], close=pred_df['Close'],
    name='XGBoost予測値', increasing_line_color='red', decreasing_line_color='blue',
    increasing_fillcolor='rgba(255, 0, 0, 0.5)', decreasing_fillcolor='rgba(0, 0, 255, 0.5)'
))

fig.update_layout(
    title=f'【データリーク排除・XGBoost単体】 Sansan(4443) 四本値のAI予測 ({base_date.strftime("%m/%d")}起点)',
    yaxis_title='株価 (円)', xaxis_rangeslider_visible=False, height=800, width=1200, template='plotly_white'
)
fig.write_html("xgboost_forecast_fixed.html")
print("\n=> チャートを xgboost_forecast_fixed.html に保存しました。")