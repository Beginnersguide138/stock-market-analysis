import pandas as pd

# Load stock data
df_stock = pd.read_csv('stock_detail_4443_20260228152816.csv', encoding='cp932')
# Filter out non-numeric prices if any
df_stock['貸借値段（円）'] = pd.to_numeric(df_stock['貸借値段（円）'], errors='coerce')
df_stock = df_stock.dropna(subset=['貸借値段（円）'])

# Date parsing
df_stock['Date'] = pd.to_datetime(df_stock['申込日'].astype(str), format='%Y%m%d')
df_stock = df_stock.sort_values('Date')

# Group by W-FRI (Weekly ending on Friday)
df_weekly = df_stock.resample('W-FRI', on='Date').last().reset_index()

# Extract Year and Month from the 'Date' to create a Week_Code
# For Week_Code, we need the week of the month.
# Let's see the week ranks
df_weekly['YearMonth'] = df_weekly['Date'].dt.strftime('%y%m')
df_weekly['WeekRank'] = df_weekly.groupby('YearMonth')['Date'].rank(method='first').astype(int)
df_weekly['Week_Code'] = df_weekly['YearMonth'].astype(str) + df_weekly['WeekRank'].astype(str).str.zfill(2)
df_weekly['Week_Code'] = df_weekly['Week_Code'].astype(int)

# Load market data
df_market = pd.read_csv('cleaned_trading_data_full_latest.csv')

# Merge
df_merged = pd.merge(df_market, df_weekly[['Week_Code', '貸借値段（円）']], on='Week_Code', how='inner')

# Calculate correlation
results = []
for market in df_merged['Market'].unique():
    for category in df_merged['Category'].unique():
        subset = df_merged[(df_merged['Market'] == market) & (df_merged['Category'] == category)]
        if len(subset) > 5:
            corr = subset['Net_Balance_Billion_JPY'].corr(subset['貸借値段（円）'])
            results.append({
                'Market': market,
                'Category': category,
                'Correlation': corr
            })

df_corr = pd.DataFrame(results).sort_values(by='Correlation', ascending=False)
print("--- 全期間 ---")
print("Top 5 Positive Correlations:")
print(df_corr.head(5))
print("\nTop 5 Negative Correlations:")
print(df_corr.tail(5))

# Filter for the last few months (e.g., Week_Code >= 251101 for Nov 2025 to Feb 2026)
df_merged_recent = df_merged[df_merged['Week_Code'] >= 251101]

results_recent = []
for market in df_merged_recent['Market'].unique():
    for category in df_merged_recent['Category'].unique():
        subset = df_merged_recent[(df_merged_recent['Market'] == market) & (df_merged_recent['Category'] == category)]
        if len(subset) > 5:
            corr = subset['Net_Balance_Billion_JPY'].corr(subset['貸借値段（円）'])
            results_recent.append({
                'Market': market,
                'Category': category,
                'Correlation': corr
            })

if results_recent:
    df_corr_recent = pd.DataFrame(results_recent).sort_values(by='Correlation', ascending=False)
    print("\n--- ここ数ヶ月 (2025年11月以降) ---")
    print("Top 5 Positive Correlations:")
    print(df_corr_recent.head(5))
    print("\nTop 5 Negative Correlations:")
    print(df_corr_recent.tail(5))
