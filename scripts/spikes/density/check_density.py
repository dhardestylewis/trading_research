import pandas as pd
df = pd.read_csv('reports/spike_tsfm/density_sweep_results.csv')
combos = df[['asset','step']].drop_duplicates().values.tolist()
print(f"Completed combos: {combos}")
for a, s in combos:
    t = df[(df['asset']==a) & (df['step']==s) & (df['fires'])]
    if len(t) > 0:
        print(f"  {a} step={int(s)}: {len(t)} trades, mean={t['net_bps'].mean():.1f}bps, total={t['net_bps'].sum():.0f}bps, win={(t['net_bps']>0).mean():.0%}")
    else:
        print(f"  {a} step={int(s)}: 0 trades")
