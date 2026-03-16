import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

COST_BPS = 14
base_dir = Path("reports")
out_dir = base_dir / "visualizations"
panel_df = pd.read_parquet("data/processed/panel_expanded/panel.parquet")

def load_and_prep(exp):
    df = pd.read_csv(base_dir / exp / "partial_results_log.csv")
    df['idx_int'] = df['timestamp'].astype(int)
    date_map = panel_df['timestamp']
    df['timestamp'] = df['idx_int'].map(date_map)
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

def get_top(df, a, m):
    cell = df[(df['asset'] == a) & (df['model'] == m)].copy()
    if cell.empty: return None
    cell['decile'] = pd.qcut(cell['predicted_move'], 10, labels=False, duplicates='drop')
    top = cell[cell['decile'] == cell['decile'].max()].copy()
    top = top.sort_values('timestamp')
    top['net_bps'] = top['realized_move_bps'] - COST_BPS
    return top

all_dfs = []

# Crypto targets
df027 = load_and_prep('exp027')
all_dfs.append(get_top(df027, 'APT-USD', 'XGBoostCounterBaseline'))
all_dfs.append(get_top(df027, 'SOL-USD', 'TabPFNTopDecile'))

df029 = load_and_prep('exp029')
all_dfs.append(get_top(df029, 'SOL-USD', 'TabPFNTopDecile'))
all_dfs.append(get_top(df029, 'SUI-USD', 'TabPFNTopDecile'))

# ALL 16 Spike targets
spike_df = pd.read_csv(base_dir / "spike_tsfm" / "lane_discovery_results_with_time.csv")
spike_df['timestamp'] = pd.to_datetime(spike_df['timestamp'], utc=True)

target_spikes = ['EWJ', 'FXA', 'FXE', 'INDA', 'UVXY', 'XLE', 'XLF', 'XLU', 'DBA']
for a in target_spikes:
    sf = spike_df[(spike_df['asset'] == a) & (spike_df['fires'])].copy()
    if not sf.empty:
        sf = sf.sort_values('timestamp')
        sf['net_bps'] = sf['pnl']
        all_dfs.append(sf)

print(f"Combining {len(all_dfs)} dataframes...")

all_trades = pd.concat(all_dfs, ignore_index=True).sort_values('timestamp')
portfolio = all_trades.groupby('timestamp').agg(net_bps=('net_bps', 'sum')).reset_index()

portfolio = portfolio.sort_values('timestamp')
portfolio['cumulative_net_pnl'] = portfolio['net_bps'].cumsum()
portfolio['drawdown'] = portfolio['cumulative_net_pnl'] - portfolio['cumulative_net_pnl'].cummax()

sns.set_theme(style="whitegrid", palette="muted")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

ax1.plot(portfolio['timestamp'], portfolio['cumulative_net_pnl'], color='orange', linewidth=2)
ax1.set_title(f"Combined Portfolio - 12 Assets (Including Marginal 'DBA')", fontsize=14, fontweight='bold')
ax1.set_ylabel("Cumulative PnL (bps)", fontsize=12)

ax2.fill_between(portfolio['timestamp'], portfolio['drawdown'], 0, color='red', alpha=0.3)
ax2.plot(portfolio['timestamp'], portfolio['drawdown'], color='red', linewidth=1)
ax2.set_ylabel("Drawdown (bps)", fontsize=12)
ax2.set_xlabel("Time", fontsize=12)

plt.tight_layout()
out_path = out_dir / "combined_portfolio_all_19_assets.png"
plt.savefig(out_path, dpi=300)
print(f"Saved {out_path}")
