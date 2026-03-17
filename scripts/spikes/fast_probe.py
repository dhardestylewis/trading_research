import pandas as pd
import numpy as np
import yfinance as yf
from tabpfn import TabPFNClassifier
import warnings
warnings.filterwarnings('ignore')

ASSET = "XLE"
MACROS = ["USO", "SPY", "^VIX", "^TNX"]

def fetch():
    data = yf.download([ASSET] + MACROS, period="3mo", interval="1h", group_by="ticker", progress=False)
    xle = data[ASSET].dropna(how="all").copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
    xle = xle.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'}).reset_index()
    if 'Date' in xle.columns: xle = xle.rename(columns={'Date':'timestamp'})
    elif 'Datetime' in xle.columns: xle = xle.rename(columns={'Datetime':'timestamp'})
    xle['timestamp'] = pd.to_datetime(xle['timestamp'], utc=True)
    xle['ret_1'] = xle['close'].pct_change()
    
    for m in MACROS:
        m_df = data[m] if isinstance(data.columns, pd.MultiIndex) else data
        m_df = m_df.dropna(how="all").reset_index()
        if 'Date' in m_df.columns: m_df = m_df.rename(columns={'Date':'timestamp'})
        elif 'Datetime' in m_df.columns: m_df = m_df.rename(columns={'Datetime':'timestamp'})
        m_df['timestamp'] = pd.to_datetime(m_df['timestamp'], utc=True)
        m_df[f'{m}_ret_1'] = m_df['Close'].pct_change()
        xle = xle.merge(m_df[['timestamp', f'{m}_ret_1']], on='timestamp', how='left')
        xle[f'{m}_ret_1'] = xle[f'{m}_ret_1'].ffill().fillna(0)
        
    xle['rv_6'] = xle['ret_1'].rolling(6).std()
    xle['fwd_ret_8'] = xle['close'].shift(-8) / xle['close'] - 1.0
    return xle.dropna().reset_index(drop=True)

df = fetch()
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

with open("results.txt", "w") as f:
    f.write(f"BASE: {ev(base)}\n")
    f.write(f"CROSS: {ev(cross)}\n")
