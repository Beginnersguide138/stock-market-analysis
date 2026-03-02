import pandas as pd
import numpy as np

# 1. Load Forex Data
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['終値'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx = df_fx.sort_values('Date')

# Calculate weekly returns (Friday to Friday)
df_fx_weekly = df_fx.resample('W-FRI', on='Date').last().reset_index()
df_fx_weekly['YearMonth'] = df_fx_weekly['Date'].dt.strftime('%y%m')
df_fx_weekly['WeekRank'] = df_fx_weekly.groupby('YearMonth')['Date'].rank(method='first').astype(int)
df_fx_weekly['Week_Code'] = df_fx_weekly['YearMonth'].astype(str) + df_fx_weekly['WeekRank'].astype(str).str.zfill(2)
df_fx_weekly['Week_Code'] = df_fx_weekly['Week_Code'].astype(int)
df_fx_weekly = df_fx_weekly.rename(columns={'終値': 'USD_JPY'})
df_fx_weekly['FX_Return'] = df_fx_weekly['USD_JPY'].pct_change()

# 2. Load Market Data
df_market = pd.read_csv('cleaned_trading_data_full_latest.csv')
df_growth_corp = df_market[(df_market['Market'] == 'TSE Growth') & (df_market['Category'] == '法　人')][['Week_Code', 'Week', 'Net_Balance_Billion_JPY']].rename(columns={'Net_Balance_Billion_JPY': 'Growth_Corp_Flow'})
df_prime_foreign = df_market[(df_market['Market'] == 'TSE Prime') & (df_market['Category'] == '海外投資家')][['Week_Code', 'Net_Balance_Billion_JPY']].rename(columns={'Net_Balance_Billion_JPY': 'Prime_Foreign_Flow'})

# 3. Merge Data
df = pd.merge(df_growth_corp, df_prime_foreign, on='Week_Code', how='inner')
df = pd.merge(df, df_fx_weekly[['Week_Code', 'USD_JPY', 'FX_Return']], on='Week_Code', how='inner')
df = df.sort_values('Week_Code').reset_index(drop=True)

# 4. Define "Peak Out" of Foreign Selling
foreign_sell_threshold = -200
recovery_threshold = 100

events = []

for i in range(1, len(df)):
    prev_foreign = df.loc[i-1, 'Prime_Foreign_Flow']
    curr_foreign = df.loc[i, 'Prime_Foreign_Flow']
    
    # Check if the previous week was a heavy selling week, and current week is a recovery
    if prev_foreign < foreign_sell_threshold and (curr_foreign - prev_foreign) > recovery_threshold:
        
        # Check future Growth_Corp_Flow
        future_growth_flows = []
        for lag in range(1, 4):
            if i + lag < len(df):
                future_growth_flows.append(df.loc[i + lag, 'Growth_Corp_Flow'])
            else:
                future_growth_flows.append(np.nan)
        
        became_buyer = any(x > 0 for x in future_growth_flows if not np.isnan(x))
        max_buy = max([x for x in future_growth_flows if not np.isnan(x)], default=0)
        
        # Capture FX context at the time of the event (Peak out week)
        fx_level = df.loc[i, 'USD_JPY']
        fx_change = df.loc[i, 'FX_Return'] # Weekly change
        
        # Determine FX trend (Yen appreciation vs depreciation)
        if fx_change > 0.005:
            fx_trend = "円安進行"
        elif fx_change < -0.005:
            fx_trend = "円高進行"
        else:
            fx_trend = "横ばい"

        events.append({
            'Week': df.loc[i, 'Week'],
            'Foreign_Event_Flow': curr_foreign,
            'USD_JPY': fx_level,
            'FX_Trend': fx_trend,
            'Did_Corp_Buy_Lag1-3': "成功(買い転換)" if became_buyer else "失敗(売り継続)",
            'Max_Buy_Lag1-3': max_buy
        })

df_events = pd.DataFrame(events)

print("=== 海外勢売りピークアウト後のグロース法人追随：為替環境による成否判定 ===\n")

print(df_events.to_string(index=False))

# 集計
print("\n--- 為替トレンド別の「先行スパン（追随買い）」成功確率 ---")
trend_counts = df_events.groupby('FX_Trend')['Did_Corp_Buy_Lag1-3'].value_counts().unstack().fillna(0)
trend_counts['勝率'] = trend_counts.get('成功(買い転換)', 0) / (trend_counts.get('成功(買い転換)', 0) + trend_counts.get('失敗(売り継続)', 0)) * 100
print(trend_counts)
