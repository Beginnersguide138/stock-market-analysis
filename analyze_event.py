import pandas as pd
import numpy as np

# Load Data
df_market = pd.read_csv('cleaned_trading_data_full_latest.csv')

# Extract necessary series
df_growth_corp = df_market[(df_market['Market'] == 'TSE Growth') & (df_market['Category'] == '法　人')][['Week_Code', 'Week', 'Net_Balance_Billion_JPY']].rename(columns={'Net_Balance_Billion_JPY': 'Growth_Corp_Flow'})
df_prime_foreign = df_market[(df_market['Market'] == 'TSE Prime') & (df_market['Category'] == '海外投資家')][['Week_Code', 'Net_Balance_Billion_JPY']].rename(columns={'Net_Balance_Billion_JPY': 'Prime_Foreign_Flow'})

# Merge and sort
df = pd.merge(df_growth_corp, df_prime_foreign, on='Week_Code', how='inner')
df = df.sort_values('Week_Code').reset_index(drop=True)

# 1. Define "Peak Out" of Foreign Selling
# A peak out is defined as a week where foreign selling was heavy (e.g. < -200 Billion JPY) 
# and the following week it recovered significantly (e.g. increased by > 100 Billion JPY).
foreign_sell_threshold = -200
recovery_threshold = 100

events = []

for i in range(1, len(df)):
    prev_foreign = df.loc[i-1, 'Prime_Foreign_Flow']
    curr_foreign = df.loc[i, 'Prime_Foreign_Flow']
    
    # Check if the previous week was a heavy selling week, and current week is a recovery
    if prev_foreign < foreign_sell_threshold and (curr_foreign - prev_foreign) > recovery_threshold:
        
        # Check what happened to Growth_Corp_Flow in the next 1 to 3 weeks
        future_growth_flows = []
        for lag in range(1, 4):
            if i + lag < len(df):
                future_growth_flows.append(df.loc[i + lag, 'Growth_Corp_Flow'])
            else:
                future_growth_flows.append(np.nan)
        
        # Determine if Growth Corp became a buyer (positive flow) in any of the next 1-3 weeks
        became_buyer = False
        max_buy = 0
        if any(x > 0 for x in future_growth_flows if not np.isnan(x)):
            became_buyer = True
            max_buy = max(x for x in future_growth_flows if not np.isnan(x))
            
        events.append({
            'Event_Week': df.loc[i, 'Week'],
            'Foreign_Sell_Prev_Week': prev_foreign,
            'Foreign_Flow_Event_Week': curr_foreign,
            'Growth_Corp_Flow_Lag1': future_growth_flows[0],
            'Growth_Corp_Flow_Lag2': future_growth_flows[1],
            'Growth_Corp_Flow_Lag3': future_growth_flows[2],
            'Did_Corp_Buy_Afterward': became_buyer,
            'Max_Buy_In_Next_3_Weeks': max_buy
        })

df_events = pd.DataFrame(events)

print("=== プライム海外勢の「売りピークアウト」を起点としたグロース法人の買い追随の検証 ===\n")
print(f"定義: 海外投資家が {foreign_sell_threshold}億円以上売り越した翌週に、{recovery_threshold}億円以上改善したタイミングを「ピークアウト」とする。\n")
print(f"抽出されたイベント回数: {len(df_events)} 回\n")

success_count = df_events['Did_Corp_Buy_Afterward'].sum()
print(f"そのうち、1〜3週間以内にグロース法人が買い越し（プラス）に転じた回数: {success_count} 回 (勝率: {success_count/len(df_events)*100:.1f}%)\n")

print("【詳細イベントリスト】")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df_events.to_string(index=False))

