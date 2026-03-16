import pandas as pd

df = pd.read_csv('reports/spike_tsfm/density_sweep_fast_results.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
xle = df[(df['asset'] == 'XLE') & (df['step'] == 1)]
xle = xle.copy()
xle['q'] = xle['timestamp'].dt.to_period('Q').astype(str)

lines = ["# XLE step=1 (every bar) — Full Quarterly Breakdown\n"]
lines.append("| Quarter | Evals | Fires | Fire% | Mean PnL | Total PnL | Win% |")
lines.append("|---------|-------|-------|-------|----------|-----------|------|")

for q in sorted(xle['q'].unique()):
    qdf = xle[xle['q'] == q]
    trades = qdf[qdf['fires']]
    n, nf = len(qdf), len(trades)
    if nf > 0:
        lines.append(f"| {q} | {n} | {nf} | {nf/n:.0%} | {trades['net_bps'].mean():.1f} | {trades['net_bps'].sum():.0f} | {(trades['net_bps']>0).mean():.0%} |")
    else:
        lines.append(f"| {q} | {n} | 0 | 0% | - | - | - |")

trades_all = xle[xle['fires']]
lines.append(f"| **TOTAL** | **{len(xle)}** | **{len(trades_all)}** | **{len(trades_all)/len(xle):.0%}** | **{trades_all['net_bps'].mean():.1f}** | **{trades_all['net_bps'].sum():.0f}** | **{(trades_all['net_bps']>0).mean():.0%}** |")

with open('reports/spike_tsfm/xle_quarterly_breakdown.md', 'w') as f:
    f.write('\n'.join(lines))
