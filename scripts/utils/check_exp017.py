import pandas as pd

df = pd.read_csv('data/processed/exp017_rv_locked/excursion_vs_realized.csv')
print("=== EXCURSION vs REALIZED ===")
for _, r in df.iterrows():
    pair = r["pair"]
    h = int(r["horizon_h"])
    exc = r["median_max_excursion_bps"]
    real = r["median_realized_net_bps"]
    cap = r["excursion_capture_ratio"]
    print(f"  {pair:10s} {h:3d}h  excursion={exc:8.1f}  realized={real:8.1f}  capture={cap:6.3f}")

print()
df2 = pd.read_csv('data/processed/exp017_rv_locked/summary_stats.csv')
print(f"Total combos: {len(df2)}")
pos = (df2["median_net_bps"] > 0).sum()
print(f"Positive median net: {pos} / {len(df2)}")
print()
print("Top 5 by median net bps:")
for _, r in df2.head(5).iterrows():
    pair = r["pair"]
    rule = r["rule"]
    st = r["spread_type"]
    h = int(r["horizon_h"])
    med = r["median_net_bps"]
    hr = r["hit_rate"]
    n = int(r["trade_count"])
    print(f"  {pair:10s} {rule:25s} {st:15s} {h:3d}h  med={med:8.1f}  hr={hr:5.3f}  n={n}")

print()
print("Bottom 5 by median net bps:")
for _, r in df2.tail(5).iterrows():
    pair = r["pair"]
    rule = r["rule"]
    st = r["spread_type"]
    h = int(r["horizon_h"])
    med = r["median_net_bps"]
    hr = r["hit_rate"]
    n = int(r["trade_count"])
    print(f"  {pair:10s} {rule:25s} {st:15s} {h:3d}h  med={med:8.1f}  hr={hr:5.3f}  n={n}")
