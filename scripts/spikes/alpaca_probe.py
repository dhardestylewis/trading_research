import os
import pandas as pd
import numpy as np
from itertools import chain
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from tabpfn import TabPFNClassifier
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY:
    print("NO ALPACA KEYS. ABORT.")
    exit(1)

api = tradeapi.REST(API_KEY, SECRET_KEY, base_url='https://paper-api.alpaca.markets', api_version='v2')

ASSET = "XLE"
MACROS = ["USO", "SPY"]

start_date = (pd.Timestamp.now(tz='America/New_York') - pd.Timedelta(days=90)).isoformat()
end_date = pd.Timestamp.now(tz='America/New_York').isoformat()

print("Fetching from Alpaca Trade API...")
try:
    bars = api.get_bars([ASSET] + MACROS, tradeapi.rest.TimeFrame.Hour, start=start_date, end=end_date, adjustment='all').df
except Exception as e:
    print(f"Alpaca fetch failed: {e}")
    exit(1)

bars = bars.reset_index()

xle = bars[bars['symbol'] == ASSET].copy()
xle = xle.rename(columns={'close': 'close'})
xle['ret_1'] = xle['close'].pct_change()

for m in MACROS:
    m_df = bars[bars['symbol'] == m].copy()
    m_df[f'{m}_ret_1'] = m_df['close'].pct_change()
    xle = xle.merge(m_df[['timestamp', f'{m}_ret_1']], on='timestamp', how='left')
    xle[f'{m}_ret_1'] = xle[f'{m}_ret_1'].ffill().fillna(0)

xle['rv_6'] = xle['ret_1'].rolling(6).std()
xle['fwd_ret_8'] = xle['close'].shift(-8) / xle['close'] - 1.0
df = xle.dropna().reset_index(drop=True)

base = ['ret_1', 'rv_6']
cross = base + [f"{m}_ret_1" for m in MACROS]

def ev(feat):
    clf = TabPFNClassifier(device='cpu')
    X_train = df[feat].iloc[-210:-10].values
    y_train = (df['fwd_ret_8'].iloc[-210:-10].values * 10000.0)
    bins = pd.qcut(y_train, 10, labels=False, duplicates='drop')
    
    X_test = df[feat].iloc[-10:].values
    y_test = df['fwd_ret_8'].iloc[-10:].values
    
    clf.fit(X_train, bins)
    p = clf.predict_proba(X_test)
    conv = p[:, -2:].sum(axis=1) if p.shape[1] >= 2 else p[:, -1]
    
    trades = 0
    pnl = 0
    for i, c in enumerate(conv):
         if c > 0.25:
             trades += 1
             pnl += y_test[i]
    return f"Trades: {trades}, PnL: {pnl*100:.3f}%"

print("Evaluating...")
if len(df) > 210:
    print("BASE:", ev(base))
    print("CROSS:", ev(cross))
else:
    print("Not enough data fetched.")
