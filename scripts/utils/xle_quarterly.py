import pandas as pd
df = pd.read_csv('reports/spike_tsfm/density_sweep_fast_results.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df['quarter'] = df['timestamp'].dt.to_period('Q').astype(str)

xle = df[(df['asset'] == 'XLE') & (df['step'] == 1)]
quarters = sorted(xle['quarter'].unique())

with open('xle_quarterly.md', 'w') as f:
    f.write("# XLE step=1 (every bar) — Quarterly Breakdown\n\n")
    f.write("| Quarter | Evals | Fires | Fire Rate | Mean PnL | Total PnL | Win Rate |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for q in quarters:
        qdf = xle[xle['quarter'] == q]
        trades = qdf[qdf['fires']]
        n = len(qdf)
        nf = len(trades)
        fr = nf / n if n > 0 else 0
        if nf > 0:
            f.write(f"| {q} | {n} | {nf} | {fr:.0%} | {trades['net_bps'].mean():.1f} | {trades['net_bps'].sum():.0f} | {(trades['net_bps']>0).mean():.0%} |\n")
        else:
            f.write(f"| {q} | {n} | 0 | 0% | - | - | - |\n")
    # Total
    trades = xle[xle['fires']]
    f.write(f"| **TOTAL** | {len(xle)} | {len(trades)} | {len(trades)/len(xle):.0%} | {trades['net_bps'].mean():.1f} | {trades['net_bps'].sum():.0f} | {(trades['net_bps']>0).mean():.0%} |\n")
