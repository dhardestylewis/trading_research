import pandas as pd
from pathlib import Path

COST_BPS = 14
SPIKE_COST_BPS = 5
base_dir = Path("reports")
panel_df = pd.read_parquet("data/processed/panel_expanded/panel.parquet")

def load_and_prep(exp):
    csv = base_dir / exp / "partial_results_log.csv"
    if not csv.exists(): return None
    df = pd.read_csv(csv)
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
    top['net_bps'] = top['realized_move_bps'] - COST_BPS
    return top[['timestamp', 'asset', 'model', 'net_bps']]

all_trades = []

# Crypto targets (3)
df027 = load_and_prep('exp027')
if df027 is not None:
    all_trades.append(get_top(df027, 'APT-USD', 'XGBoostCounterBaseline'))
    all_trades.append(get_top(df027, 'SOL-USD', 'TabPFNTopDecile'))

df029 = load_and_prep('exp029')
if df029 is not None:
    all_trades.append(get_top(df029, 'SOL-USD', 'TabPFNTopDecile'))
    all_trades.append(get_top(df029, 'SUI-USD', 'TabPFNTopDecile')) # Note SUI starts later usually

# Spike targets (8)
spike_df = pd.read_csv(base_dir / "spike_tsfm" / "lane_discovery_results_with_time.csv")
spike_df['timestamp'] = pd.to_datetime(spike_df['timestamp'], utc=True)
target_spikes = ['EWJ', 'FXA', 'FXE', 'INDA', 'UVXY', 'XLE', 'XLF', 'XLU']
for a in target_spikes:
    sf = spike_df[(spike_df['asset'] == a) & (spike_df['fires'])].copy()
    if not sf.empty:
        sf['net_bps'] = sf['pnl']  # already has SPIKE_COST_BPS (5) applied
        sf['model'] = 'SpikeTabPFN'
        all_trades.append(sf[['timestamp', 'asset', 'model', 'net_bps']])

# Combine
combined = pd.concat([df for df in all_trades if df is not None], ignore_index=True)
combined['year_month'] = combined['timestamp'].dt.to_period('M')
combined['quarter'] = combined['timestamp'].dt.to_period('Q')

print("\n--- PnL by Quarter ---")
q_summary = combined.groupby('quarter').agg(
    trades=('net_bps', 'count'),
    net_pnl=('net_bps', 'sum'),
    win_rate=('net_bps', lambda x: (x > 0).mean())
)
print(q_summary)

print("\n--- PnL by Month (2024) ---")
m24 = combined[combined['timestamp'].dt.year == 2024]
if not m24.empty:
    m_summary = m24.groupby('year_month').agg(
        trades=('net_bps', 'count'),
        net_pnl=('net_bps', 'sum'),
        win_rate=('net_bps', lambda x: (x > 0).mean())
    )
    print(m_summary)

print("\n--- PnL by Asset Context (Pre and Post Jan 2024) ---")
pre_2024 = combined[combined['timestamp'] < '2024-01-01']
post_jan_2024 = combined[combined['timestamp'] > '2024-01-31']

print(f"Total PnL Pre-2024:    {pre_2024['net_bps'].sum():.1f} bps (n={len(pre_2024)})")
print(f"Total PnL Jan 2024:    {combined[(combined['timestamp'].dt.year == 2024) & (combined['timestamp'].dt.month == 1)]['net_bps'].sum():.1f} bps")
print(f"Total PnL Feb onwards: {post_jan_2024['net_bps'].sum():.1f} bps (n={len(post_jan_2024)})")
