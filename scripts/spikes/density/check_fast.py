import pandas as pd
from pathlib import Path
p = Path('reports/spike_tsfm/density_sweep_fast_results.csv')
if not p.exists():
    print("Fast results not written yet")
else:
    df = pd.read_csv(p)
    combos = df[['asset','step']].drop_duplicates().values.tolist()
    print(f"FAST: {len(combos)} / 10 combos done")
    for a, s in sorted(combos, key=lambda x: (x[0], x[1])):
        t = df[(df['asset']==a) & (df['step']==s) & (df['fires'])]
        n = len(df[(df['asset']==a) & (df['step']==s)])
        if len(t) > 0:
            print(f"  {a} step={int(s)}: {n} evals, {len(t)} trades, mean={t['net_bps'].mean():.1f}bps, total={t['net_bps'].sum():.0f}bps, win={(t['net_bps']>0).mean():.0%}")
        else:
            print(f"  {a} step={int(s)}: {n} evals, 0 trades")
