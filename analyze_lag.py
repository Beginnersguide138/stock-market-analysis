import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Load Market Data
df_market = pd.read_csv('cleaned_trading_data_full_latest.csv')
# Keep only necessary data to speed up
df_growth_corp = df_market[(df_market['Market'] == 'TSE Growth') & (df_market['Category'] == '法　人')].copy()
df_growth_corp = df_growth_corp[['Week_Code', 'Week', 'Net_Balance_Billion_JPY']].rename(columns={'Net_Balance_Billion_JPY': 'Growth_Corp_Flow'})

# Other potentially leading actors
df_prime_foreign = df_market[(df_market['Market'] == 'TSE Prime') & (df_market['Category'] == '海外投資家')][['Week_Code', 'Net_Balance_Billion_JPY']].rename(columns={'Net_Balance_Billion_JPY': 'Prime_Foreign_Flow'})
df_prime_corp = df_market[(df_market['Market'] == 'TSE Prime') & (df_market['Category'] == '法　人')][['Week_Code', 'Net_Balance_Billion_JPY']].rename(columns={'Net_Balance_Billion_JPY': 'Prime_Corp_Flow'})
df_growth_trust = df_market[(df_market['Market'] == 'TSE Growth') & (df_market['Category'] == '投資信託')][['Week_Code', 'Net_Balance_Billion_JPY']].rename(columns={'Net_Balance_Billion_JPY': 'Growth_Trust_Flow'})

# 2. Load Sansan Stock Data
df_stock = pd.read_csv('stock_detail_4443_20260228152816.csv', encoding='cp932')
df_stock['貸借値段（円）'] = pd.to_numeric(df_stock['貸借値段（円）'], errors='coerce')
df_stock = df_stock.dropna(subset=['貸借値段（円）'])
df_stock['Date'] = pd.to_datetime(df_stock['申込日'].astype(str), format='%Y%m%d')
df_stock = df_stock.sort_values('Date')
df_weekly = df_stock.resample('W-FRI', on='Date').last().reset_index()
df_weekly['YearMonth'] = df_weekly['Date'].dt.strftime('%y%m')
df_weekly['WeekRank'] = df_weekly.groupby('YearMonth')['Date'].rank(method='first').astype(int)
df_weekly['Week_Code'] = df_weekly['YearMonth'].astype(str) + df_weekly['WeekRank'].astype(str).str.zfill(2)
df_weekly['Week_Code'] = df_weekly['Week_Code'].astype(int)

# 3. Merge All Data
df = pd.merge(df_growth_corp, df_prime_foreign, on='Week_Code', how='left')
df = pd.merge(df, df_prime_corp, on='Week_Code', how='left')
df = pd.merge(df, df_growth_trust, on='Week_Code', how='left')
df = pd.merge(df, df_weekly[['Week_Code', 'Date', '貸借値段（円）']], on='Week_Code', how='left')

df = df.sort_values('Week_Code').reset_index(drop=True)

# 4. Cross-Correlation Analysis (Lead/Lag)
# We want to see if Shifted(Actor) correlates with Current(Growth_Corp)
# Shift 1 means Actor moved 1 week BEFORE Growth Corp
lags = [1, 2, 3, 4]
actors_to_test = ['Prime_Foreign_Flow', 'Prime_Corp_Flow', 'Growth_Trust_Flow', '貸借値段（円）']

print("=== クロス相関分析: グロース法人買いに先行する指標は何か？ ===")
for actor in actors_to_test:
    print(f"\n[{actor} の先行性]")
    # lag 0 (同週)
    corr_0 = df['Growth_Corp_Flow'].corr(df[actor])
    print(f"  Lag 0 (同週): {corr_0:.3f}")
    
    for lag in lags:
        # Shift the actor forward by 'lag' weeks (so we compare past actor with current Growth Corp)
        shifted_series = df[actor].shift(lag)
        corr_lag = df['Growth_Corp_Flow'].corr(shifted_series)
        print(f"  Lag {lag} (指標が{lag}週先行): {corr_lag:.3f}")

# Generate a visual comparison of Prime Foreign vs Growth Corp with a 1-2 week shift
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=df['Week'], y=df['Prime_Foreign_Flow'], name='プライム海外投資家 (左軸)', opacity=0.3, marker_color='blue'), secondary_y=False)
fig.add_trace(go.Scatter(x=df['Week'], y=df['Growth_Corp_Flow'], name='グロース法人 (右軸/スケール小)', line=dict(color='red', width=3)), secondary_y=True)

fig.update_layout(title="プライム海外勢の動向 vs グロース法人の動向 (先行性の視覚的確認)", hovermode='x unified')
fig.write_html("leading_indicator_analysis.html")
