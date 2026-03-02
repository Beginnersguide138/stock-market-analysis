import pandas as pd

# Load stock data
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

# Load market data
df_market = pd.read_csv('cleaned_trading_data_full_latest.csv')

# Merge
df_merged = pd.merge(df_market, df_weekly[['Week_Code', 'Date', '貸借値段（円）']], on='Week_Code', how='inner')

# Filter the last 5 weeks
recent_weeks = sorted(df_merged['Week_Code'].unique())[-5:]
df_recent = df_merged[df_merged['Week_Code'].isin(recent_weeks)]

# Extract specific categories
target_categories = [
    ('TSE Prime', '海外投資家'),
    ('TSE Prime', '法　人'),
    ('TSE Prime', '投資信託'),
    ('TSE Growth', '法　人')
]

print("Recent 5 Weeks Analysis:")
for w in recent_weeks:
    subset = df_recent[df_recent['Week_Code'] == w]
    if subset.empty: continue
    
    date_str = subset['Date'].iloc[0].strftime('%Y-%m-%d')
    price = subset['貸借値段（円）'].iloc[0]
    week_str = subset['Week'].iloc[0]
    
    print(f"\nWeek: {week_str} (Code: {w}, Date ending: {date_str}) | Sansan Price: {price} JPY")
    for market, cat in target_categories:
        val = subset[(subset['Market'] == market) & (subset['Category'] == cat)]['Net_Balance_Billion_JPY']
        if not val.empty:
            print(f"  {market} - {cat}: {val.iloc[0]:.2f} Billion JPY")

