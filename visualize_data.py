import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# データの読み込み
df = pd.read_csv('cleaned_trading_data_full_latest.csv')
df = df.sort_values(by=['Week_Code'])

# 各市場・投資部門ごとの累積を計算
df['Cumulative_Balance_Billion_JPY'] = df.groupby(['Market', 'Category'])['Net_Balance_Billion_JPY'].cumsum()

# 市場とカテゴリの一覧を取得
markets = df['Market'].unique()
categories = df['Category'].unique()

# カラーパレットの割り当て
colors = px.colors.qualitative.Plotly
color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

# サブプロットの作成 (市場ごとに縦に並べる)
fig = make_subplots(
    rows=len(markets), cols=1, 
    subplot_titles=markets,
    shared_xaxes=True,
    vertical_spacing=0.03
)

# 各市場ごとにデータをプロット
for i, market in enumerate(markets, 1):
    market_df = df[df['Market'] == market]
    
    for category in categories:
        cat_df = market_df[market_df['Category'] == category]
        if cat_df.empty:
            continue
            
        # 週次の増減（棒グラフ）
        fig.add_trace(
            go.Bar(
                x=cat_df['Week'],
                y=cat_df['Net_Balance_Billion_JPY'],
                name=f"{category} (週次)",
                legendgroup=f"bar_{category}",
                marker_color=color_map[category],
                opacity=0.4, # 棒グラフは少し透過させて折れ線を見やすくする
                showlegend=(i == 1), # 凡例は最初のサブプロットのみ表示
                hovertemplate=f"部門: {category}<br>週次: %{{y:.2f}} 10億円<extra></extra>"
            ),
            row=i, col=1
        )
        
        # 累積（折れ線グラフ）
        fig.add_trace(
            go.Scatter(
                x=cat_df['Week'],
                y=cat_df['Cumulative_Balance_Billion_JPY'],
                name=f"{category} (累計)",
                legendgroup=f"line_{category}",
                line=dict(color=color_map[category], width=3),
                mode='lines+markers',
                showlegend=(i == 1),
                hovertemplate=f"部門: {category}<br>累計: %{{y:.2f}} 10億円<extra></extra>"
            ),
            row=i, col=1
        )

# 各サブプロットにゼロの基準線を追加
for i in range(1, len(markets) + 1):
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=i, col=1)

# レイアウトの全体調整
fig.update_layout(
    barmode='group', # 棒グラフを横に並べる
    height=3200,
    width=2400,
    hovermode="x unified",
    title="投資部門別 株式売買状況 (週次増減[棒] と 累計[折れ線] / 10億円)",
    margin=dict(l=50, r=50, t=100, b=100)
)

# X軸・Y軸のラベル表示や角度調整
fig.update_xaxes(tickangle=45, showticklabels=True)
fig.update_yaxes(title_text="買越/売越 (10億円)", matches=None, showticklabels=True)

# サブプロットのタイトル（市場名）のフォントサイズを大きくする
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(size=24)

# HTMLとして保存
output_file = 'trading_data_visualization.html'
fig.write_html(output_file)
print(f"Visualization saved to {output_file}")
