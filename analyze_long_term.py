import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load and prepare data
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

df_market = pd.read_csv('cleaned_trading_data_full_latest.csv')
df_merged = pd.merge(df_market, df_weekly[['Week_Code', 'Date', '貸借値段（円）']], on='Week_Code', how='inner')

# Target actors in TSE Prime
actors = ['海外投資家', '法　人', '投資信託', '個　人']
df_prime = df_merged[df_merged['Market'] == 'TSE Prime']

# Create pivot table for actors
df_pivot = df_prime.pivot_table(index=['Date', '貸借値段（円）'], columns='Category', values='Net_Balance_Billion_JPY').reset_index()
df_pivot = df_pivot.sort_values('Date')

# Calculate cumulative sums for better trend visualization
for actor in actors:
    if actor in df_pivot.columns:
        df_pivot[f'{actor}_CumSum'] = df_pivot[actor].cumsum()

# Plotting
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, 
                    subplot_titles=('Sansan (4443) 株価推移', 'プライム市場 主要アクターの売買代金 (累計 / 10億円)'),
                    row_heights=[0.4, 0.6])

# 1. Stock Price
fig.add_trace(go.Scatter(x=df_pivot['Date'], y=df_pivot['貸借値段（円）'], 
                         mode='lines+markers', name='株価', line=dict(color='black', width=2)),
              row=1, col=1)

# 2. Actors Cumulative Sum
colors = {'海外投資家': 'blue', '法　人': 'red', '投資信託': 'green', '個　人': 'orange'}
for actor in actors:
    if f'{actor}_CumSum' in df_pivot.columns:
        fig.add_trace(go.Scatter(x=df_pivot['Date'], y=df_pivot[f'{actor}_CumSum'],
                                 mode='lines', name=f'{actor} (累計)', 
                                 line=dict(color=colors.get(actor, 'gray'), width=2)),
                      row=2, col=1)

fig.update_layout(height=1000, width=1200, hovermode='x unified', title='Sansan株価と主要アクター(プライム)の長期動向')
fig.add_hline(y=0, line_dash="dash", row=2, col=1)

output_file = 'long_term_analysis.html'
fig.write_html(output_file)
print(f"Analysis saved to {output_file}")
