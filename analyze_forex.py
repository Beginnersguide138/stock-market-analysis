import pandas as pd

# Load Forex Data
df_fx = pd.read_csv('forex-data.csv')
# 複数ファイルの結合などでヘッダー行がデータ内に混ざっている場合を除去
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['終値'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx = df_fx.sort_values('Date')
# Resample to weekly (ending on Friday)
df_fx_weekly = df_fx.resample('W-FRI', on='Date').last().reset_index()

# Extract Year and Month from the 'Date' to create a Week_Code
df_fx_weekly['YearMonth'] = df_fx_weekly['Date'].dt.strftime('%y%m')
df_fx_weekly['WeekRank'] = df_fx_weekly.groupby('YearMonth')['Date'].rank(method='first').astype(int)
df_fx_weekly['Week_Code'] = df_fx_weekly['YearMonth'].astype(str) + df_fx_weekly['WeekRank'].astype(str).str.zfill(2)
df_fx_weekly['Week_Code'] = df_fx_weekly['Week_Code'].astype(int)
df_fx_weekly = df_fx_weekly.rename(columns={'終値': 'USD_JPY'})

# Load Stock Data
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

# Load Market Data
df_market = pd.read_csv('cleaned_trading_data_full_latest.csv')

# Merge
df_merged = pd.merge(df_market, df_weekly[['Week_Code', '貸借値段（円）']], on='Week_Code', how='inner')
df_merged = pd.merge(df_merged, df_fx_weekly[['Week_Code', 'USD_JPY']], on='Week_Code', how='inner')

print("=== 為替（USD/JPY）と各アクター・株価の相関分析 ===")

# 株価との相関
corr_stock_fx = df_merged['貸借値段（円）'].corr(df_merged['USD_JPY'])
print(f"\nSansan株価 と USD/JPY の相関: {corr_stock_fx:.3f}")

# プライム市場の海外投資家との相関
df_prime_foreign = df_merged[(df_merged['Market'] == 'TSE Prime') & (df_merged['Category'] == '海外投資家')]
corr_foreign_fx = df_prime_foreign['Net_Balance_Billion_JPY'].corr(df_prime_foreign['USD_JPY'])
print(f"プライム市場 海外投資家売買代金 と USD/JPY の相関: {corr_foreign_fx:.3f}")

# グロース法人の買いとの相関
df_growth_corp = df_merged[(df_merged['Market'] == 'TSE Growth') & (df_merged['Category'] == '法　人')]
corr_growth_corp_fx = df_growth_corp['Net_Balance_Billion_JPY'].corr(df_growth_corp['USD_JPY'])
print(f"グロース市場 法人売買代金 と USD/JPY の相関: {corr_growth_corp_fx:.3f}")

# 直近3ヶ月
print("\n--- 直近3ヶ月 (2025年11月以降) ---")
df_recent = df_merged[df_merged['Week_Code'] >= 251101]

if not df_recent.empty:
    corr_stock_fx_recent = df_recent['貸借値段（円）'].corr(df_recent['USD_JPY'])
    print(f"Sansan株価 と USD/JPY の相関: {corr_stock_fx_recent:.3f}")
    
    df_prime_foreign_recent = df_recent[(df_recent['Market'] == 'TSE Prime') & (df_recent['Category'] == '海外投資家')]
    if len(df_prime_foreign_recent) > 1:
        corr_foreign_fx_recent = df_prime_foreign_recent['Net_Balance_Billion_JPY'].corr(df_prime_foreign_recent['USD_JPY'])
        print(f"プライム市場 海外投資家売買代金 と USD/JPY の相関: {corr_foreign_fx_recent:.3f}")

    df_growth_corp_recent = df_recent[(df_recent['Market'] == 'TSE Growth') & (df_recent['Category'] == '法　人')]
    if len(df_growth_corp_recent) > 1:
        corr_growth_corp_fx_recent = df_growth_corp_recent['Net_Balance_Billion_JPY'].corr(df_growth_corp_recent['USD_JPY'])
        print(f"グロース市場 法人売買代金 と USD/JPY の相関: {corr_growth_corp_fx_recent:.3f}")

