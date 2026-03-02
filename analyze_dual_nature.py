import pandas as pd
import numpy as np

# データの読み込み
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

# 変化率（リターン）を計算
df_weekly['Stock_Return'] = df_weekly['貸借値段（円）'].pct_change()

df_market = pd.read_csv('cleaned_trading_data_full_latest.csv')
df_merged = pd.merge(df_market, df_weekly[['Week_Code', 'Date', '貸借値段（円）', 'Stock_Return']], on='Week_Code', how='inner')

# プライム市場とグロース市場の主要アクターのデータを抽出
target_markets = ['TSE Prime', 'TSE Growth']
target_categories = ['海外投資家', '法　人', '投資信託', '個　人']

print("=== プライム市場 vs グロース市場 アクター別相関分析 ===")

# 全期間の価格との相関
print("\n【株価そのものとの相関 (全期間)】")
for market in target_markets:
    print(f"--- {market} ---")
    for category in target_categories:
        subset = df_merged[(df_merged['Market'] == market) & (df_merged['Category'] == category)]
        corr = subset['Net_Balance_Billion_JPY'].corr(subset['貸借値段（円）'])
        print(f"  {category}: {corr:.3f}")

# 株価の「変化率（リターン）」と売買代金の相関（より本質的な値動きの連動性）
print("\n【株価の変化率（リターン）との相関 (全期間)】")
for market in target_markets:
    print(f"--- {market} ---")
    for category in target_categories:
        subset = df_merged[(df_merged['Market'] == market) & (df_merged['Category'] == category)].copy()
        subset = subset.dropna(subset=['Stock_Return'])
        corr = subset['Net_Balance_Billion_JPY'].corr(subset['Stock_Return'])
        print(f"  {category}: {corr:.3f}")

# 直近3ヶ月（パニック相場期：Week_Code >= 251101）の価格との相関
print("\n【株価そのものとの相関 (直近3ヶ月: 2025年11月以降)】")
df_recent = df_merged[df_merged['Week_Code'] >= 251101]
for market in target_markets:
    print(f"--- {market} ---")
    for category in target_categories:
        subset = df_recent[(df_recent['Market'] == market) & (df_recent['Category'] == category)]
        if len(subset) > 3:
            corr = subset['Net_Balance_Billion_JPY'].corr(subset['貸借値段（円）'])
            print(f"  {category}: {corr:.3f}")

