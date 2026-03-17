import pandas as pd
import numpy as np
import yfinance as yf
from tabpfn import TabPFNClassifier
import concurrent.futures

WINDOW = 150

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
        
    print("BASE:", evaluate(df, baseline))
    print("CROSS:", evaluate(df, cross))
