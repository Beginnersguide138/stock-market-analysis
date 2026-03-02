import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. データ読み込みと前処理
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['終値'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx = df_fx.sort_values('Date')
df_fx_weekly = df_fx.resample('W-FRI', on='Date').last().reset_index()
df_fx_weekly['YearMonth'] = df_fx_weekly['Date'].dt.strftime('%y%m')
df_fx_weekly['WeekRank'] = df_fx_weekly.groupby('YearMonth')['Date'].rank(method='first').astype(int)
df_fx_weekly['Week_Code'] = df_fx_weekly['YearMonth'].astype(str) + df_fx_weekly['WeekRank'].astype(str).str.zfill(2)
df_fx_weekly['Week_Code'] = df_fx_weekly['Week_Code'].astype(int)
df_fx_weekly = df_fx_weekly.rename(columns={'終値': 'USD_JPY'})

df_stock = pd.read_csv('stock_detail_4443_20260228152816.csv', encoding='cp932')
df_stock['貸借値段（円）'] = pd.to_numeric(df_stock['貸借値段（円）'], errors='coerce')
df_stock = df_stock.dropna(subset=['貸借値段（円）'])
df_stock['Date_Stock'] = pd.to_datetime(df_stock['申込日'].astype(str), format='%Y%m%d')
df_stock = df_stock.sort_values('Date_Stock')
df_weekly = df_stock.resample('W-FRI', on='Date_Stock').last().reset_index()
df_weekly['YearMonth'] = df_weekly['Date_Stock'].dt.strftime('%y%m')
df_weekly['WeekRank'] = df_weekly.groupby('YearMonth')['Date_Stock'].rank(method='first').astype(int)
df_weekly['Week_Code'] = df_weekly['YearMonth'].astype(str) + df_weekly['WeekRank'].astype(str).str.zfill(2)
df_weekly['Week_Code'] = df_weekly['Week_Code'].astype(int)
df_weekly = df_weekly.rename(columns={'貸借値段（円）': 'Sansan_Price'})

df_market = pd.read_csv('cleaned_trading_data_full_latest.csv')
df_growth_corp = df_market[(df_market['Market'] == 'TSE Growth') & (df_market['Category'] == '法　人')][['Week_Code', 'Net_Balance_Billion_JPY']].rename(columns={'Net_Balance_Billion_JPY': 'Growth_Corp_Flow'})
df_prime_foreign = df_market[(df_market['Market'] == 'TSE Prime') & (df_market['Category'] == '海外投資家')][['Week_Code', 'Net_Balance_Billion_JPY']].rename(columns={'Net_Balance_Billion_JPY': 'Prime_Foreign_Flow'})

# 統合
df = pd.merge(df_growth_corp, df_prime_foreign, on='Week_Code', how='inner')
df = pd.merge(df, df_fx_weekly[['Week_Code', 'Date', 'USD_JPY']], on='Week_Code', how='inner')
df = pd.merge(df, df_weekly[['Week_Code', 'Sansan_Price']], on='Week_Code', how='inner')
df = df.sort_values('Date').reset_index(drop=True)

# 2. 特徴量エンジニアリング（過去4週間のラグ特徴量を作成）
target_fx = 'USD_JPY'
target_stock = 'Sansan_Price'

for i in range(1, 5):
    df[f'USD_JPY_lag_{i}'] = df['USD_JPY'].shift(i)
    df[f'Sansan_Price_lag_{i}'] = df['Sansan_Price'].shift(i)
    df[f'Prime_Foreign_Flow_lag_{i}'] = df['Prime_Foreign_Flow'].shift(i)
    df[f'Growth_Corp_Flow_lag_{i}'] = df['Growth_Corp_Flow'].shift(i)

df = df.dropna().reset_index(drop=True)
features = [col for col in df.columns if 'lag' in col]

X = df[features]
y_fx = df[target_fx]
y_stock = df[target_stock]

# 3. XGBoostモデルの学習
model_fx = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model_stock = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

model_fx.fit(X, y_fx)
model_stock.fit(X, y_stock)

# 4. 今後8週間（約2ヶ月）の再帰的予測
future_dates = [df['Date'].iloc[-1] + pd.Timedelta(weeks=i) for i in range(1, 9)]
predictions_fx = []
predictions_stock = []

last_row = df.iloc[-1].copy()
current_features = {f: last_row[f] for f in features}

for i in range(8):
    X_pred = pd.DataFrame([current_features])[features]
    pred_fx = model_fx.predict(X_pred)[0]
    pred_stock = model_stock.predict(X_pred)[0]
    
    predictions_fx.append(pred_fx)
    predictions_stock.append(pred_stock)
    
    next_features = {}
    for j in range(4, 1, -1):
        next_features[f'USD_JPY_lag_{j}'] = current_features[f'USD_JPY_lag_{j-1}']
        next_features[f'Sansan_Price_lag_{j}'] = current_features[f'Sansan_Price_lag_{j-1}']
        next_features[f'Prime_Foreign_Flow_lag_{j}'] = current_features[f'Prime_Foreign_Flow_lag_{j-1}']
        next_features[f'Growth_Corp_Flow_lag_{j}'] = current_features[f'Growth_Corp_Flow_lag_{j-1}']
        
    next_features['USD_JPY_lag_1'] = pred_fx
    next_features['Sansan_Price_lag_1'] = pred_stock
    # 未来の投資家動向はわからないため、徐々にゼロ（中立）へ減衰させると仮定
    next_features['Prime_Foreign_Flow_lag_1'] = current_features['Prime_Foreign_Flow_lag_1'] * 0.5 
    next_features['Growth_Corp_Flow_lag_1'] = current_features['Growth_Corp_Flow_lag_1'] * 0.5 
    
    current_features = next_features

df_forecast = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
    'USD_JPY_Forecast': np.round(predictions_fx, 2),
    'Sansan_Price_Forecast': np.round(predictions_stock, 0)
})

print("=== 機械学習(XGBoost)による今後8週間の予測結果 ===")
print(df_forecast.to_string(index=False))

# 5. 結果の可視化
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df['Date'], y=df['USD_JPY'], name='USD/JPY (実績)', line=dict(color='blue')), secondary_y=False)
fig.add_trace(go.Scatter(x=future_dates, y=predictions_fx, name='USD/JPY (予測)', line=dict(color='blue', dash='dash')), secondary_y=False)

fig.add_trace(go.Scatter(x=df['Date'], y=df['Sansan_Price'], name='Sansan株価 (実績)', line=dict(color='red')), secondary_y=True)
fig.add_trace(go.Scatter(x=future_dates, y=predictions_stock, name='Sansan株価 (予測)', line=dict(color='red', dash='dash')), secondary_y=True)

fig.update_layout(title="為替(USD/JPY)とSansan株価の機械学習予測 (今後8週間)", hovermode='x unified')
fig.update_yaxes(title_text="USD/JPY", secondary_y=False)
fig.update_yaxes(title_text="Sansan 株価 (円)", secondary_y=True)
fig.write_html("ml_forecast.html")
