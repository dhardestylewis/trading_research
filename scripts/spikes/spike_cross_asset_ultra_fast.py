import pandas as pd
import numpy as np
import yfinance as yf
from tabpfn import TabPFNClassifier
import warnings
warnings.filterwarnings("ignore")

WINDOW = 200  # Smaller window
ASSET = "XLE"
MACROS = ["USO", "SPY", "^VIX", "^TNX"]
ALL_TICKERS = [ASSET] + MACROS

def fetch_and_merge() -> pd.DataFrame:
    data = yf.download(ALL_TICKERS, period="3mo", interval="1h", group_by="ticker", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        xle_df = data[ASSET].dropna(how="all").copy()
    else:
        xle_df = data.copy()
        
    xle_df = xle_df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
    xle_df.index.name = "timestamp"
    xle_df = xle_df.reset_index()
    xle_df["timestamp"] = pd.to_datetime(xle_df["timestamp"], utc=True)
    xle_df["ret_1"] = xle_df["close"].pct_change()
    
    for macro in MACROS:
        m_df = data[macro] if isinstance(data.columns, pd.MultiIndex) else data
        m_df = m_df.dropna(how="all").reset_index()
        m_df["timestamp"] = pd.to_datetime(m_df.iloc[:,0], utc=True)
        m_df[f"{macro}_ret_1"] = m_df["Close"].pct_change()
        m_df[f"{macro}_ret_3"] = m_df["Close"].pct_change(3)
        m_df[f"{macro}_rv_6"] = m_df[f"{macro}_ret_1"].rolling(6).std()
        
        cols = ["timestamp", f"{macro}_ret_1", f"{macro}_ret_3", f"{macro}_rv_6"]
        xle_df = xle_df.merge(m_df[cols], on="timestamp", how="left")
        
        xle_df[f"{macro}_ret_1"] = xle_df[f"{macro}_ret_1"].ffill().fillna(0.0)
        xle_df[f"{macro}_ret_3"] = xle_df[f"{macro}_ret_3"].ffill().fillna(0.0)
        xle_df[f"{macro}_rv_6"] = xle_df[f"{macro}_rv_6"].ffill().fillna(0.0)
        
        xle_df[f"resid_{macro}"] = xle_df["ret_1"] - xle_df[f"{macro}_ret_1"]
        xle_df[f"beta_proxy_{macro}"] = xle_df["ret_1"].rolling(24).cov(xle_df[f"{macro}_ret_1"]) / (xle_df[f"{macro}_ret_1"].rolling(24).var() + 1e-8)
        
    xle_df["rv_6"] = xle_df["ret_1"].rolling(6).std()
    xle_df["fwd_ret_8"] = xle_df["close"].shift(-8) / xle_df["close"] - 1.0
    xle_df["target_bps"] = xle_df["fwd_ret_8"] * 10000.0
    
    return xle_df.dropna().reset_index(drop=True)

def evaluate(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    profits = []
    trades = 0
    # Evaluate ONLY the last 50 logical 1h intervals for absolute speed
    eval_start = max(WINDOW, len(df) - 58) 
    
    for i in range(eval_start, len(df)-8, 10):
        train = df.iloc[i-WINDOW:i]
        test = df.iloc[i:i+10]
        
        X_train = train[feat_cols].values
        X_test = test[feat_cols].values
        y_train = train["target_bps"].values
        
        bins = pd.qcut(y_train, 10, labels=False, duplicates='drop')
        clf = TabPFNClassifier()
        clf.fit(X_train, bins)
        
        probas = clf.predict_proba(X_test)
        conv = probas[:, -2:].sum(axis=1) if probas.shape[1] >= 2 else probas[:, -1]
            
        for j, c in enumerate(conv):
            if c > 0.25:
                trades += 1
                profits.append(test["fwd_ret_8"].iloc[j] - 0.0005)
                
    net = sum(profits) if profits else 0.0
    win_rate = sum(p > 0 for p in profits) / len(profits) if trades > 0 else 0
    return {"trades": trades, "net_pct": net * 100, "win_rate": win_rate * 100}

if __name__ == "__main__":
    df = fetch_and_merge()
    
    baseline = ["ret_1", "rv_6"]
    cross = baseline.copy()
    for m in MACROS:
        cross.extend([f"{m}_ret_1", f"{m}_ret_3", f"{m}_rv_6", f"resid_{m}", f"beta_proxy_{m}"])
        
    b_res = evaluate(df, baseline)
    c_res = evaluate(df, cross)
    print("BASE:", b_res)
    print("CROSS:", c_res)
