import pandas as pd
import numpy as np
import yfinance as yf
from tabpfn import TabPFNClassifier
import concurrent.futures
import json

WINDOW = 150
ASSET = "XLE"
MACROS = ["USO", "SPY", "^VIX", "^TNX"]
ALL_TICKERS = [ASSET] + MACROS

def fetch_and_merge() -> pd.DataFrame:
    data = yf.download(ALL_TICKERS, period="2y", interval="1h", group_by="ticker", progress=False)
    xle_df = data[ASSET].dropna(how="all").copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
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

def eval_window(args):
    train, test, feat_cols = args
    X_train = train[feat_cols].values
    X_test = test[feat_cols].values
    y_train = train["target_bps"].values
    bins = pd.qcut(y_train, 10, labels=False, duplicates='drop')
    
    clf = TabPFNClassifier()
    clf.fit(X_train, bins)
    probas = clf.predict_proba(X_test)
    conv = probas[:, -2:].sum(axis=1) if probas.shape[1] >= 2 else probas[:, -1]
    
    trades = []
    for j, c in enumerate(conv):
        if c > 0.25:
            ts = test["timestamp"].iloc[j].strftime("%Y-%m-%d %H:%M")
            pnl = float(test["fwd_ret_8"].iloc[j] - 0.0005) * 100.0
            trades.append({"timestamp": ts, "conviction": round(c, 3), "pnl_pct": round(pnl, 2)})
            
    return trades

def evaluate(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    args_list = []
    # Step by 40 to evaluate dramatically faster while preserving broad market coverage length
    for i in range(WINDOW, len(df)-8, 40):
        args_list.append((df.iloc[i-WINDOW:i], df.iloc[i:i+40], feat_cols))
        
    all_trades = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for t in executor.map(eval_window, args_list):
            all_trades.extend(t)
            
    net = sum(x["pnl_pct"] for x in all_trades) if all_trades else 0.0
    win_rate = sum(x["pnl_pct"] > 0 for x in all_trades) / len(all_trades) if all_trades else 0
    return {"trades": len(all_trades), "net_pct": round(net, 2), "win_rate": round(win_rate * 100, 2), "ledger": all_trades}

if __name__ == "__main__":
    df = fetch_and_merge()
    print(df.shape)
    
    baseline = ["ret_1", "rv_6"]
    cross = baseline.copy()
    for m in MACROS:
        cross.extend([f"{m}_ret_1", f"{m}_ret_3", f"{m}_rv_6", f"resid_{m}", f"beta_proxy_{m}"])
        
    base_res = evaluate(df, baseline)
    print("BASE:", {k: v for k, v in base_res.items() if k != "ledger"})
    
    cross_res = evaluate(df, cross)
    print("CROSS:", {k: v for k, v in cross_res.items() if k != "ledger"})
    
    with open("ledger.json", "w") as f:
        json.dump({"BASE": base_res, "CROSS": cross_res}, f, indent=4)
