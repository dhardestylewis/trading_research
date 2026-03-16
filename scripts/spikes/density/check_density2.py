import pandas as pd
df = pd.read_csv('reports/spike_tsfm/density_sweep_results.csv')
combos = df[['asset','step']].drop_duplicates().values.tolist()
with open('density_live.md', 'w') as f:
    f.write(f"Completed: {len(combos)} / 10 combos\n\n")
    f.write("| Asset | Step | Trades | Mean PnL | Total PnL | Win Rate |\n")
    f.write("|---|---|---|---|---|---|\n")
    for a, s in sorted(combos, key=lambda x: (x[0], x[1])):
        t = df[(df['asset']==a) & (df['step']==s) & (df['fires'])]
        if len(t) > 0:
            f.write(f"| {a} | {int(s)} | {len(t)} | {t['net_bps'].mean():.1f} | {t['net_bps'].sum():.0f} | {(t['net_bps']>0).mean():.0%} |\n")
        else:
            f.write(f"| {a} | {int(s)} | 0 | - | - | - |\n")
