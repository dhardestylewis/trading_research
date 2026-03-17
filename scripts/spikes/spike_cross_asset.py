import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_score
from tabpfn import TabPFNClassifier
from pathlib import Path

WINDOW = 500
ASSET = "XLE"
MACROS = ["USO", "SPY", "^VIX", "^TNX"]
ALL_TICKERS = [ASSET] + MACROS

def fetch_and_merge() -> pd.DataFrame:
    print(f"Fetching 2y hourly data for {ALL_TICKERS}...")
    data = yf.download(ALL_TICKERS, period="2y", interval="1h", group_by="ticker", progress=False)
    
    # Extract asset
    if isinstance(data.columns, pd.MultiIndex):
        xle_df = data[ASSET].dropna(how="all").copy()
    else:
        xle_df = data.copy()
        
    xle_df = xle_df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
    xle_df.index.name = "timestamp"
    xle_df = xle_df.reset_index()
    xle_df["timestamp"] = pd.to_datetime(xle_df["timestamp"], utc=True)
    xle_df["ret_1"] = xle_df["close"].pct_change()
    
    # Merge exogenous
    for macro in MACROS:
        m_df = data[macro] if isinstance(data.columns, pd.MultiIndex) else data
        m_df = m_df.dropna(how="all")
        m_df = m_df.reset_index()
        m_df["timestamp"] = pd.to_datetime(m_df.iloc[:,0], utc=True)
        
        # Calculate trailing features for macro
        m_df[f"{macro}_ret_1"] = m_df["Close"].pct_change()
        m_df[f"{macro}_ret_3"] = m_df["Close"].pct_change(3)
        m_df[f"{macro}_rv_6"] = m_df[f"{macro}_ret_1"].rolling(6).std()
        
        # Inner Join cleanly to XLE
        cols = ["timestamp", f"{macro}_ret_1", f"{macro}_ret_3", f"{macro}_rv_6"]
        xle_df = xle_df.merge(m_df[cols], on="timestamp", how="left")
        
        # Forward fill macroeconomic NA values that drift out of sync with equity market hours
        xle_df[f"{macro}_ret_1"] = xle_df[f"{macro}_ret_1"].ffill().fillna(0.0)
        xle_df[f"{macro}_ret_3"] = xle_df[f"{macro}_ret_3"].ffill().fillna(0.0)
        xle_df[f"{macro}_rv_6"] = xle_df[f"{macro}_rv_6"].ffill().fillna(0.0)
        
        # Cross-asset relative features (Beta proxies)
        xle_df[f"resid_{macro}"] = xle_df["ret_1"] - xle_df[f"{macro}_ret_1"]
        xle_df[f"beta_proxy_{macro}"] = xle_df["ret_1"].rolling(24).cov(xle_df[f"{macro}_ret_1"]) / (xle_df[f"{macro}_ret_1"].rolling(24).var() + 1e-8)
        
    # XLE native features
    xle_df["rv_6"] = xle_df["ret_1"].rolling(6).std()
    xle_df["fwd_ret_8"] = xle_df["close"].shift(-8) / xle_df["close"] - 1.0
    xle_df["target_bps"] = xle_df["fwd_ret_8"] * 10000.0
    
    return xle_df.dropna().reset_index(drop=True)


def evaluate(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    profits = []
    trades = 0
    
    # Walk forward exactly like canary
    for i in range(WINDOW, len(df)-8, 20):  # Stride 20 for speed (1-week steps)
        train = df.iloc[i-WINDOW:i]
        test = df.iloc[i:i+20]
        
        X_train = train[feat_cols].values
        X_test = test[feat_cols].values
        
        y_train = train["target_bps"].values
        bins = pd.qcut(y_train, 10, labels=False, duplicates='drop')
        
        clf = TabPFNClassifier()
        clf.fit(X_train, bins)
        
        probas = clf.predict_proba(X_test)
        if probas.shape[1] >= 2:
            conv = probas[:, -2:].sum(axis=1)
        else:
            conv = probas[:, -1]
            
        for j, c in enumerate(conv):
            if c > 0.25:
                trades += 1
                # Deduct 5bps slip
                profit = test["fwd_ret_8"].iloc[j] - 0.0005
                profits.append(profit)
                
    net = sum(profits) if profits else 0.0
    win_rate = sum(p > 0 for p in profits) / len(profits) if trades > 0 else 0
    return {"trades": trades, "net_pct": net * 100, "win_rate": win_rate * 100}

if __name__ == "__main__":
    df = fetch_and_merge()
    print(f"Constructed grand matrix shape: {df.shape}")
    
    # Baseline Features
    baseline_feats = ["ret_1", "rv_6"]
    
    # Cross-Asset Features
    cross_feats = baseline_feats.copy()
    for m in MACROS:
        cross_feats.extend([f"{m}_ret_1", f"{m}_ret_3", f"{m}_rv_6", f"resid_{m}", f"beta_proxy_{m}"])
        
    print("\n--- BASELINE (Univariate) ---")
    res_base = evaluate(df, baseline_feats)
    print(res_base)
    
    print("\n--- CROSS ASSET (Multivariate) ---")
    res_cross = evaluate(df, cross_feats)
    print(res_cross)
