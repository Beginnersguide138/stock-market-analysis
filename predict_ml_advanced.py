import pandas as pd
import numpy as np
import xgboost as xgb

# 1. データ読み込みと前処理
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['終値'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx = df_fx.sort_values('Date')
df_fx_weekly = df_fx.resample('W-FRI', on='Date').last().reset_index()

# 特徴量: 為替のモメンタム（変化率）
df_fx_weekly['USD_JPY_Return_1w'] = df_fx_weekly['終値'].pct_change(1)
df_fx_weekly['USD_JPY_Return_4w'] = df_fx_weekly['終値'].pct_change(4)

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

# 特徴量: 株価のモメンタム
df_weekly['Stock_Return_1w'] = df_weekly['貸借値段（円）'].pct_change(1)

df_weekly['YearMonth'] = df_weekly['Date_Stock'].dt.strftime('%y%m')
df_weekly['WeekRank'] = df_weekly.groupby('YearMonth')['Date_Stock'].rank(method='first').astype(int)
df_weekly['Week_Code'] = df_weekly['YearMonth'].astype(str) + df_weekly['WeekRank'].astype(str).str.zfill(2)
df_weekly['Week_Code'] = df_weekly['Week_Code'].astype(int)
df_weekly = df_weekly.rename(columns={'貸借値段（円）': 'Sansan_Price'})


df_market = pd.read_csv('cleaned_trading_data_full_latest.csv')

# ピボットしてすべてのアクターを特徴量として持つ
df_market_pivot = df_market.pivot_table(
    index='Week_Code', 
    columns=['Market', 'Category'], 
    values='Net_Balance_Billion_JPY', 
    fill_value=0
)
# 列名をフラットに ('TSE Prime', '海外投資家') -> 'TSE_Prime_海外投資家'
df_market_pivot.columns = [f"{c[0].replace(' ', '_')}_{c[1]}" for c in df_market_pivot.columns]
df_market_pivot = df_market_pivot.reset_index()

# 統合
df = pd.merge(df_market_pivot, df_fx_weekly[['Week_Code', 'Date', 'USD_JPY', 'USD_JPY_Return_1w', 'USD_JPY_Return_4w']], on='Week_Code', how='inner')
df = pd.merge(df, df_weekly[['Week_Code', 'Sansan_Price', 'Stock_Return_1w']], on='Week_Code', how='inner')
df = df.sort_values('Date').reset_index(drop=True)

# 2. 特徴量エンジニアリング（ラグ特徴量）
# NOTE: This script uses only lag features (shift(i) with i>0), so there is no data leakage
# from future data in feature engineering. Lag features are computed on the full dataset
# but only reference past values, which is safe.
target_stock = 'Sansan_Price'
# 使用するベース特徴量
base_features = [c for c in df.columns if c not in ['Date', 'Week_Code', 'Sansan_Price']]

# 過去4週分のラグを生成
for i in range(1, 5):
    for f in base_features + [target_stock]:
        df[f'{f}_lag_{i}'] = df[f].shift(i)

df = df.dropna().reset_index(drop=True)
features = [col for col in df.columns if 'lag_' in col]

X = df[features]
y_stock = df[target_stock]

# 3. XGBoostモデルの学習と特徴量重要度の抽出
model_stock = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
model_stock.fit(X, y_stock)

# 特徴量重要度を取得して上位10個を表示
importance = model_stock.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("=== 機械学習モデル: どの特徴量がSansanの株価予測に効いたか（重要度トップ10） ===")
print(feature_importance_df.head(10).to_string(index=False))

# 4. 今後8週間（約2ヶ月）の再帰的予測
future_dates = [df['Date'].iloc[-1] + pd.Timedelta(weeks=i) for i in range(1, 9)]
predictions_stock = []

last_row = df.iloc[-1].copy()
current_features = {f: last_row[f] for f in features}

for i in range(8):
    X_pred = pd.DataFrame([current_features])[features]
    pred_stock = model_stock.predict(X_pred)[0]
    predictions_stock.append(pred_stock)
    
    # 次のステップのための特徴量更新
    next_features = {}
    for f in base_features + [target_stock]:
        for j in range(4, 1, -1):
            next_features[f'{f}_lag_{j}'] = current_features[f'{f}_lag_{j-1}']
    
    next_features['Sansan_Price_lag_1'] = pred_stock
    
    # 為替は横ばい（Return=0）と仮定、投資家の売買動向は0に減衰すると仮定
    for f in base_features:
        if 'USD_JPY' in f and 'Return' not in f:
            next_features[f'{f}_lag_1'] = current_features[f'{f}_lag_1'] # 為替レートは据え置き
        else:
            next_features[f'{f}_lag_1'] = current_features[f'{f}_lag_1'] * 0.5 # 減衰
            
    current_features = next_features

df_forecast = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
    'Sansan_Price_Forecast': np.round(predictions_stock, 0)
})

print("\n=== より多くの特徴量を用いた今後8週間の予測結果 ===")
print(df_forecast.to_string(index=False))

# --- Cross Validation with TimeSeriesSplit ---
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n=== TimeSeriesSplit Cross-Validation (5-fold) ===")
tscv = TimeSeriesSplit(n_splits=5)
mae_scores = []
rmse_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
    y_t, y_v = y_stock.iloc[train_idx], y_stock.iloc[val_idx]
    model_cv = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model_cv.fit(X_t, y_t)
    pred = model_cv.predict(X_v)
    mae_scores.append(mean_absolute_error(y_v, pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_v, pred)))
    print(f"  Sansan_Price Fold {fold}: MAE={mae_scores[-1]:.4f}, RMSE={rmse_scores[-1]:.4f}")
print(f"  Sansan_Price Avg MAE: {np.mean(mae_scores):.4f}, Avg RMSE: {np.mean(rmse_scores):.4f}")
