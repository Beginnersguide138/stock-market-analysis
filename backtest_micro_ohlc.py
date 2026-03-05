import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
import xgboost as xgb
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# 1. データの取得
print("Fetching daily data for 4443.T (Sansan), ^DJI (Dow Jones), and ^N225 (Nikkei 225) from Yahoo Finance...")
df_sansan = yf.download('4443.T', period='2y')
df_dji = yf.download('^DJI', period='2y')
df_n225 = yf.download('^N225', period='2y')

if isinstance(df_sansan.columns, pd.MultiIndex):
    df_sansan.columns = df_sansan.columns.get_level_values(0)
if isinstance(df_dji.columns, pd.MultiIndex):
    df_dji.columns = df_dji.columns.get_level_values(0)
if isinstance(df_n225.columns, pd.MultiIndex):
    df_n225.columns = df_n225.columns.get_level_values(0)

df_sansan = df_sansan.reset_index()
df_dji = df_dji.reset_index()
df_n225 = df_n225.reset_index()

# Note: ffill deferred to after train/test split to prevent data leakage

# 2. 指数・為替の特徴量作成
df_dji['DJI_Close'] = df_dji['Close']
df_dji['DJI_Return'] = df_dji['Close'].pct_change()
df_dji['Date_JP'] = df_dji['Date'] + pd.Timedelta(days=1)
df_dji['Date_JP'] = df_dji['Date_JP'].apply(lambda x: x + pd.Timedelta(days=2) if x.weekday() == 5 else (x + pd.Timedelta(days=1) if x.weekday() == 6 else x))

df_n225['N225_Close'] = df_n225['Close']
df_n225['N225_Return'] = df_n225['Close'].pct_change()

# 3. テクニカル指標の計算関数 (データリーク防止: 必要な範囲のみで計算)
def compute_features(df_input):
    """テクニカル指標を計算する。入力dfのコピーに対して計算を行う。"""
    df_out = df_input.copy()

    ichi = IchimokuIndicator(high=df_out['High'], low=df_out['Low'], window1=9, window2=26, window3=52)
    df_out['Ichi_Tenkan'] = ichi.ichimoku_conversion_line()
    df_out['Ichi_Kijun'] = ichi.ichimoku_base_line()
    df_out['Ichi_SpanA'] = ichi.ichimoku_a()
    df_out['Ichi_SpanB'] = ichi.ichimoku_b()
    df_out['Close_lag26'] = df_out['Close'].shift(26)

    df_out['RSI'] = RSIIndicator(close=df_out['Close'], window=14).rsi()
    macd = MACD(close=df_out['Close'])
    df_out['MACD'] = macd.macd()
    df_out['MACD_Signal'] = macd.macd_signal()
    df_out['MACD_Hist'] = macd.macd_diff()

    stoch = StochasticOscillator(high=df_out['High'], low=df_out['Low'], close=df_out['Close'], window=14, smooth_window=3)
    df_out['Stoch_K'] = stoch.stoch()
    df_out['Stoch_D'] = stoch.stoch_signal()

    df_out['EMA_12'] = df_out['Close'].ewm(span=12, adjust=False).mean()
    df_out['EMA_26'] = df_out['Close'].ewm(span=26, adjust=False).mean()

    df_out['Return'] = df_out['Close'].pct_change()
    df_out['Vol_Change'] = df_out['Volume'].pct_change()

    return df_out

# 4. データ結合 (外部データのmergeは生データに対して行う - テクニカル指標は分割後に計算)
df_raw = df_sansan.copy()

df_raw = pd.merge(df_raw, df_dji[['Date_JP', 'DJI_Close', 'DJI_Return']], left_on='Date', right_on='Date_JP', how='left')
df_raw['DJI_Return'] = df_raw['DJI_Return'].fillna(0)

df_raw = pd.merge(df_raw, df_n225[['Date', 'N225_Close', 'N225_Return']], on='Date', how='left')
df_raw['N225_Return'] = df_raw['N225_Return'].fillna(0)

# 為替データ(USD/JPY)
df_fx = pd.read_csv('forex-data.csv')
df_fx = df_fx[df_fx['日付'] != '日付'].dropna(subset=['日付'])
df_fx['Date'] = pd.to_datetime(df_fx['日付'], format='%y/%m/%d')
df_fx['USD_JPY'] = pd.to_numeric(df_fx['終値'], errors='coerce')
df_fx['USD_JPY_Return'] = df_fx['USD_JPY'].pct_change(-1)

df_raw = pd.merge(df_raw, df_fx[['Date', 'USD_JPY', 'USD_JPY_Return']], on='Date', how='left')
# Note: USD_JPY ffill deferred to after train/test split
df_raw['USD_JPY_Return'].fillna(0, inplace=True)

# 5. 四本値のターゲット作成
targets = ['Open', 'High', 'Low', 'Close']
for t in targets:
    for i in range(1, 4): # 2/26, 2/27, (2/28は土曜なので3/2) を予測する
        df_raw[f'Target_{t}_{i}d'] = df_raw[t].shift(-i)

features = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Vol_Change',
    'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB', 'Close_lag26',
    'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
    'DJI_Return', 'N225_Return', 'USD_JPY', 'USD_JPY_Return'
]

# === バックテスト: 2/25（水）までのデータで、2/26, 2/27, 3/2の四本値を予測 ===
# データリーク防止: 基準日 + 予測horizon分までの生データのみでテクニカル指標を計算
base_date_str = '2026-02-25'
cutoff_date = pd.to_datetime('2026-03-02')  # 3日先まで予測するので3/2まで
df_cutoff = df_raw[df_raw['Date'] <= cutoff_date].copy()
df_all = compute_features(df_cutoff)
df_all = df_all.dropna().reset_index(drop=True)

test_date_start = pd.to_datetime('2026-02-26')
gap = pd.Timedelta(days=1)
df_train = df_all[df_all['Date'] < test_date_start - gap].copy()
df_train.ffill(inplace=True)

if df_train.empty:
    print("Error: 訓練データが空です。")
    exit()

base_data = df_train.iloc[-1].copy()
base_date = base_data['Date'].strftime('%Y-%m-%d')
print(f"\n=== 四本値(OHLC) バックテスト実行 (基準日: {base_date}, 終値: {base_data['Close']} 円) ===")

X_train = df_train[features]
X_test = pd.DataFrame([base_data[features]])

predictions = {}

for t in targets:
    predictions[t] = []
    for i in range(1, 4):
        model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(X_train, df_train[f'Target_{t}_{i}d'])
        pred = model.predict(X_test)[0]
        predictions[t].append(pred)

# 論理的整合性（Low <= Open/Close <= High）の補正
final_preds = []
for i in range(3):
    p_open = predictions['Open'][i]
    p_high = predictions['High'][i]
    p_low = predictions['Low'][i]
    p_close = predictions['Close'][i]
    
    act_high = max(p_high, p_open, p_close, p_low)
    act_low = min(p_low, p_open, p_close, p_high)
    
    final_preds.append({
        'Open': np.round(p_open, 0),
        'High': np.round(act_high, 0),
        'Low': np.round(act_low, 0),
        'Close': np.round(p_close, 0)
    })

# 実績値の取得 (YFinanceから直で取得して比較)
actual_df = yf.download('4443.T', start='2026-02-26', end='2026-03-03')
if isinstance(actual_df.columns, pd.MultiIndex):
    actual_df.columns = actual_df.columns.get_level_values(0)
actual_df = actual_df.reset_index()
actual_df['Date_str'] = actual_df['Date'].dt.strftime('%Y-%m-%d')

actual_dates = ['2026-02-26', '2026-02-27', '2026-03-02']

for i, date_str in enumerate(actual_dates):
    actual = actual_df[actual_df['Date_str'] == date_str]
    if not actual.empty:
        a_open = actual['Open'].values[0]
        a_high = actual['High'].values[0]
        a_low = actual['Low'].values[0]
        a_close = actual['Close'].values[0]
    else:
        a_open = a_high = a_low = a_close = None
        
    p = final_preds[i]
    
    print(f"\n【{date_str} の予測と実績】")
    print(f"       [ 始値(Open) | 高値(High) | 安値(Low) | 終値(Close) ]")
    print(f"  予測: {p['Open']:>10.0f} | {p['High']:>10.0f} | {p['Low']:>9.0f} | {p['Close']:>11.0f}")
    
    if a_open is not None:
        print(f"  実績: {a_open:>10.0f} | {a_high:>10.0f} | {a_low:>9.0f} | {a_close:>11.0f}")
        print(f"  誤差: {abs(p['Open']-a_open):>10.0f} | {abs(p['High']-a_high):>10.0f} | {abs(p['Low']-a_low):>9.0f} | {abs(p['Close']-a_close):>11.0f}")
    else:
        print("  実績: データ未確定")
