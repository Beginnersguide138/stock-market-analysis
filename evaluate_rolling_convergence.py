import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import xgboost as xgb
import plotly.graph_objects as go
from datetime import timedelta
import jpholiday
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import sys
from tqdm import tqdm

ticker = sys.argv[1] if len(sys.argv) > 1 else '6526.T'
lookback_days = int(sys.argv[2]) if len(sys.argv) > 2 else 120 # Default to ~6 months

print(f"Fetching daily data for {ticker}...")
# Fetch 3 years of data to ensure we have a full 1-year training window even for the oldest simulation day
df_target = yf.download(ticker, period='3y', progress=False) 
df_dji = yf.download('^DJI', period='3y', progress=False)
df_n225 = yf.download('^N225', period='3y', progress=False)
df_vix = yf.download('^VIX', period='3y', progress=False)
df_topix = yf.download('1306.T', period='3y', progress=False)
df_growth = yf.download('2516.T', period='3y', progress=False)

for df in [df_target, df_dji, df_n225, df_vix, df_topix, df_growth]:
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.ffill(inplace=True)

df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
for df_us in [df_dji, df_vix]:
    df_us['Date_JP'] = df_us['Date'] + pd.Timedelta(days=1)
    df_us['Date_JP'] = df_us['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

df_n225['N225_Return'] = df_n225['Close'].pct_change()
df_vix['VIX_Close'] = df_vix['Close']
df_vix['VIX_Return'] = df_vix['Close'].pct_change()
df_topix['TOPIX_Return'] = df_topix['Close'].pct_change()
df_growth['Growth_Return'] = df_growth['Close'].pct_change()

df_target['Date'] = pd.to_datetime(df_target['Date']).dt.tz_localize(None)
for df_us in [df_dji, df_vix]: df_us['Date_JP'] = pd.to_datetime(df_us['Date_JP']).dt.tz_localize(None)
for df_jp in [df_n225, df_topix, df_growth]: df_jp['Date'] = pd.to_datetime(df_jp['Date']).dt.tz_localize(None)

df = pd.merge(df_target, df_dji[['Date_JP', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df = pd.merge(df, df_vix[['Date_JP', 'VIX_Close', 'VIX_Return']], left_on='Date', right_on='Date_JP', how='left', suffixes=('', '_vix'))
for col in ['DJI_Return', 'VIX_Close', 'VIX_Return']: df[col] = df[col].ffill().fillna(0)

df = pd.merge(df, df_n225[['Date', 'N225_Return']], on='Date', how='left')
df = pd.merge(df, df_topix[['Date', 'TOPIX_Return']], on='Date', how='left')
df = pd.merge(df, df_growth[['Date', 'Growth_Return']], on='Date', how='left')
for col in ['N225_Return', 'TOPIX_Return', 'Growth_Return']: df[col] = df[col].fillna(0)

df['USD_JPY_Return_1d_ahead'] = 0.0

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
df['Day_of_Week'] = df['Date'].dt.dayofweek

base_features = [
    'Return', 'Vol_Change', 'RSI', 'MACD_Ratio', 'MACD_Hist_Ratio',
    'ATR_Ratio', 'BB_Width_Ratio', 'BB_Pos',
    'Upper_Shadow_Ratio', 'Lower_Shadow_Ratio',
    'Upper_Shadow_5d_MA', 'Lower_Shadow_5d_MA',
    'DJI_Return', 'N225_Return', 'USD_JPY_Return_1d_ahead',
    'VIX_Close', 'VIX_Return', 'TOPIX_Return', 'Growth_Return', 'Day_of_Week'
]

for i in range(1, 6):
    df[f'Target_High_{i}d'] = (df['High'].shift(-i) - df['Open'].shift(-i)) / df['Open'].shift(-i)
    df[f'Target_Low_{i}d']  = (df['Low'].shift(-i) - df['Open'].shift(-i)) / df['Open'].shift(-i)
    df[f'Target_Open_Gap_{i}d'] = (df['Open'].shift(-i) - df['Close'].shift(-i+1)) / df['Close'].shift(-i+1)

df = df.replace([np.inf, -np.inf], np.nan)

all_predictions = []
tscv = TimeSeriesSplit(n_splits=2)
# Keep params light for a 120-day loop (120 retrains * 15 models = 1800 fits)
param_grid = {'max_depth': [3], 'learning_rate': [0.05], 'n_estimators': [100]} 

print(f"\nRunning True Rolling Training & Prediction for {lookback_days} days...")
for step in tqdm(range(lookback_days, 0, -1)):
    current_idx = len(df) - step - 1
    if current_idx < 250: continue # Need at least ~1 year of history for the first simulation
    
    df_train_full = df.iloc[:current_idx+1].copy()
    # Use only the last 250 rows (approx 1 year) for training to mimic the original script's "short memory" philosophy
    df_train = df_train_full.iloc[-250:].dropna(subset=base_features + [f'Target_High_{i}d' for i in range(1,6)] + [f'Target_Low_{i}d' for i in range(1,6)])
    X_train = df_train[base_features]
    
    models = {}
    for i in range(1, 6):
        grid_high = GridSearchCV(xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.90, random_state=42, tree_method='hist'), param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_high.fit(X_train, df_train[f'Target_High_{i}d'])
        models[f'High_{i}d'] = grid_high.best_estimator_
        
        grid_low = GridSearchCV(xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.10, random_state=42, tree_method='hist'), param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_low.fit(X_train, df_train[f'Target_Low_{i}d'])
        models[f'Low_{i}d'] = grid_low.best_estimator_
        
        grid_open = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42, tree_method='hist'), param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_open.fit(X_train, df_train[f'Target_Open_Gap_{i}d'])
        models[f'Open_Gap_{i}d'] = grid_open.best_estimator_

    last_row = df.iloc[current_idx - 1].copy()
    last_row.fillna(0, inplace=True)
    X_pred = pd.DataFrame([last_row[base_features]])
    
    origin_date = df.iloc[current_idx]['Date']
    prev_close = last_row['Close']
    
    next_days = []
    tmp_date = origin_date
    while len(next_days) < 5:
        tmp_date += timedelta(days=1)
        if tmp_date.weekday() < 5 and not jpholiday.is_holiday(tmp_date):
            next_days.append(tmp_date)

    for idx, target_date in enumerate(next_days):
        i = idx + 1
        pred_open_gap = models[f'Open_Gap_{i}d'].predict(X_pred)[0]
        pred_open = prev_close * (1 + pred_open_gap)
        pred_high = pred_open * (1 + max(models[f'High_{i}d'].predict(X_pred)[0], 0.0))
        pred_low = pred_open * (1 + min(models[f'Low_{i}d'].predict(X_pred)[0], 0.0))
        
        all_predictions.append({
            'Origin_Date': origin_date,
            'Target_Date': target_date,
            'Days_Ahead': i,
            'Pred_Low': pred_low,
            'Pred_Open': pred_open,
            'Pred_High': pred_high,
            'Origin_Days_Ago': step
        })
        prev_close = (pred_open + pred_high + pred_low) / 3.0

df_preds = pd.DataFrame(all_predictions)
unique_targets = sorted(df_preds['Target_Date'].unique())

convergence_results = []
for tgt in unique_targets:
    preds_for_tgt = df_preds[df_preds['Target_Date'] == tgt]
    if len(preds_for_tgt) >= 3:
        std_high = preds_for_tgt['Pred_High'].std()
        std_low = preds_for_tgt['Pred_Low'].std()
        mean_price = preds_for_tgt['Pred_Open'].mean()
        volatility_spread = ((std_high + std_low) / 2) / mean_price * 100
        cv_score = 100 - (volatility_spread * 20) 
        
        convergence_results.append({
            'Target_Date': tgt,
            'Num_Viewpoints': len(preds_for_tgt),
            'Mean_Pred_High': preds_for_tgt['Pred_High'].mean(),
            'Mean_Pred_Low': preds_for_tgt['Pred_Low'].mean(),
            'Std_High': std_high,
            'Std_Low': std_low,
            'Confidence_Score': cv_score
        })

df_conv = pd.DataFrame(convergence_results)

from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3],
    subplot_titles=(f"Prediction Convergence & Actuals ({ticker} - Last {lookback_days} days)", "Confidence Score (Higher = More Reliable)")
)

df_plot = df.iloc[-lookback_days - 15:]
fig.add_trace(go.Candlestick(
    x=df_plot['Date'], open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'],
    name='Actual Price', increasing_line_color='black', decreasing_line_color='black'
), row=1, col=1)

for tgt in unique_targets:
    preds = df_preds[df_preds['Target_Date'] == tgt]
    if len(preds) > 0:
        fig.add_trace(go.Box(
            x=preds['Target_Date'], y=preds['Pred_High'],
            name='High Convergence', marker_color='rgba(255, 0, 0, 0.6)', 
            boxpoints='all', jitter=0.5, pointpos=-1.8, width=0.4, showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Box(
            x=preds['Target_Date'], y=preds['Pred_Low'],
            name='Low Convergence', marker_color='rgba(0, 0, 255, 0.6)',
            boxpoints='all', jitter=0.5, pointpos=-1.8, width=0.4, showlegend=False
        ), row=1, col=1)

if not df_conv.empty:
    colors = ['green' if val > 0 else 'red' for val in df_conv['Confidence_Score']]
    fig.add_trace(go.Bar(
        x=df_conv['Target_Date'], y=df_conv['Confidence_Score'],
        name='Confidence Score', marker_color=colors
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

fig.update_layout(
    height=900, width=1200, template='plotly_white',
    xaxis_rangeslider_visible=False,
    showlegend=False,
    hovermode="x unified"
)
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"convergence_{ticker.replace('.T', '')}_{lookback_days}days_{timestamp}.html"
fig.write_html(filename)
print(f"\nGenerated visually enhanced report: {filename}")