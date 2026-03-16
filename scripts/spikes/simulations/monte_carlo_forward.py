"""
monte_carlo_forward.py — Forward simulation using the backtest's empirical return distribution.

Samples from the actual realized PnL distribution of XLE step=1 trades to generate
N possible future equity paths over the next quarter. Produces confidence bands.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the actual backtest trade returns
df = pd.read_csv('reports/spike_tsfm/density_sweep_fast_results.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
xle = df[(df['asset'] == 'XLE') & (df['step'] == 1)].copy()

# Empirical distribution of trade returns (only when model fires)
fired = xle[xle['fires']]
empirical_returns_bps = fired['net_bps'].values

# Empirical fire rate
fire_rate = len(fired) / len(xle)

# Stats from the backtest
print(f"=== Empirical Backtest Stats (XLE step=1) ===")
print(f"Total bars evaluated: {len(xle)}")
print(f"Total fires: {len(fired)}")
print(f"Fire rate: {fire_rate:.1%}")
print(f"Mean return per trade: {empirical_returns_bps.mean():.1f} bps")
print(f"Std return per trade: {empirical_returns_bps.std():.1f} bps")
print(f"Win rate: {(empirical_returns_bps > 0).mean():.1%}")
print(f"Median: {np.median(empirical_returns_bps):.1f} bps")
print(f"Skew: {pd.Series(empirical_returns_bps).skew():.2f}")
print()

# Simulation parameters
STARTING_CAPITAL = 1000.0  # dollars
HOURS_PER_QUARTER = 63 * 7  # ~63 trading days × 6.5 market hours, rounded to 7
MAX_CONCURRENT = 4  # max overlapping 8-hour positions
CAPITAL_PER_TRADE = STARTING_CAPITAL / MAX_CONCURRENT
N_SIMULATIONS = 10000
HORIZON_HOURS = HOURS_PER_QUARTER

np.random.seed(42)

# Run Monte Carlo
final_balances = []
all_paths = []

for sim in range(N_SIMULATIONS):
    balance = STARTING_CAPITAL
    path = [balance]
    
    for hour in range(HORIZON_HOURS):
        # Each hour: does the model fire?
        if np.random.random() < fire_rate:
            # Sample a return from the empirical distribution
            ret_bps = np.random.choice(empirical_returns_bps)
            # Convert bps to dollar gain on the per-trade capital allocation
            dollar_gain = (ret_bps / 10000) * (balance / MAX_CONCURRENT)
            balance += dollar_gain
        path.append(balance)
    
    final_balances.append(balance)
    if sim < 500:  # store first 500 paths for plotting
        all_paths.append(path)

final_balances = np.array(final_balances)
all_paths = np.array(all_paths)

# Compute percentiles
p5 = np.percentile(final_balances, 5)
p25 = np.percentile(final_balances, 25)
p50 = np.percentile(final_balances, 50)
p75 = np.percentile(final_balances, 75)
p95 = np.percentile(final_balances, 95)

print(f"=== Monte Carlo Forward Projection: {N_SIMULATIONS} simulations ===")
print(f"Starting capital: ${STARTING_CAPITAL:.0f}")
print(f"Horizon: 1 quarter (~{HORIZON_HOURS} market hours)")
print(f"Max concurrent positions: {MAX_CONCURRENT}")
print()
print(f"  5th percentile (worst case):  ${p5:.0f}  ({(p5/STARTING_CAPITAL - 1)*100:+.1f}%)")
print(f" 25th percentile:               ${p25:.0f}  ({(p25/STARTING_CAPITAL - 1)*100:+.1f}%)")
print(f" 50th percentile (median):      ${p50:.0f}  ({(p50/STARTING_CAPITAL - 1)*100:+.1f}%)")
print(f" 75th percentile:               ${p75:.0f}  ({(p75/STARTING_CAPITAL - 1)*100:+.1f}%)")
print(f" 95th percentile (best case):   ${p95:.0f}  ({(p95/STARTING_CAPITAL - 1)*100:+.1f}%)")
print(f" Probability of loss:           {(final_balances < STARTING_CAPITAL).mean():.1%}")

# Plot 1: Fan chart of equity paths
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Path percentiles for fan chart
hours = np.arange(all_paths.shape[1])
p5_path = np.percentile(all_paths, 5, axis=0)
p25_path = np.percentile(all_paths, 25, axis=0)
p50_path = np.percentile(all_paths, 50, axis=0)
p75_path = np.percentile(all_paths, 75, axis=0)
p95_path = np.percentile(all_paths, 95, axis=0)

ax = axes[0]
ax.fill_between(hours, p5_path, p95_path, alpha=0.15, color='steelblue', label='5th-95th %ile')
ax.fill_between(hours, p25_path, p75_path, alpha=0.3, color='steelblue', label='25th-75th %ile')
ax.plot(hours, p50_path, color='steelblue', linewidth=2, label='Median')
ax.axhline(y=STARTING_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Market Hours', fontsize=12)
ax.set_ylabel('Portfolio Value ($)', fontsize=12)
ax.set_title('XLE TabPFN Strategy: 1-Quarter Forward Projection\n$1,000 starting capital, 10,000 simulations', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Distribution of final balances
ax2 = axes[1]
ax2.hist(final_balances, bins=80, color='steelblue', edgecolor='white', alpha=0.7)
ax2.axvline(x=STARTING_CAPITAL, color='red', linestyle='--', label=f'Breakeven (${STARTING_CAPITAL:.0f})', linewidth=2)
ax2.axvline(x=p50, color='green', linestyle='--', label=f'Median (${p50:.0f})', linewidth=2)
ax2.set_xlabel('Final Portfolio Value ($)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title(f'Distribution of Outcomes\nP(loss)={((final_balances < STARTING_CAPITAL).mean())*100:.0f}%', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = Path('reports/visualizations/monte_carlo_xle_forward.png')
plt.savefig(out, dpi=300)
print(f"\nSaved to {out}")
