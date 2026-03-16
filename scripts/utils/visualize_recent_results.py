"""
Visualize the top decile predictions for recent experiments (exp027, exp029), 
plotting the cumulative PnL, drawdowns, and distribution of realized returns.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("visualize_recent_results")

COST_BPS = 14

def load_and_prep_data(experiment_dir: str):
    csv_path = Path(experiment_dir) / "partial_results_log.csv"
    if not csv_path.exists():
        logger.warning(f"File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    if df['timestamp'].dtype in ['int64', 'float64']:
        # We need a synthetic timeline to plot line charts correctly
        df['timestamp'] = pd.date_range('2022-01-01', periods=len(df), freq='h')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    return df

def process_top_decile(df: pd.DataFrame, asset: str, model: str):
    # Isolate cell
    cell = df[(df['asset'] == asset) & (df['model'] == model)].copy()
    if cell.empty:
        return None
        
    # Get top decile of predicted move
    cell['decile'] = pd.qcut(cell['predicted_move'], 10, labels=False, duplicates='drop')
    top = cell[cell['decile'] == cell['decile'].max()].copy()
    
    top = top.sort_values('timestamp')
    top['net_bps'] = top['realized_move_bps'] - COST_BPS
    top['cumulative_net_pnl'] = top['net_bps'].cumsum()
    
    # Calculate drawdown
    running_max = top['cumulative_net_pnl'].cummax()
    top['drawdown'] = top['cumulative_net_pnl'] - running_max
    
    return top

def create_visualizations(top_df: pd.DataFrame, title_prefix: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Plot 1: Cumulative PnL & Drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(top_df['timestamp'], top_df['cumulative_net_pnl'], color='blue', linewidth=2)
    ax1.set_title(f"{title_prefix} - Cumulative Net PnL (Top Decile, {COST_BPS}bps cost)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative PnL (bps)", fontsize=12)
    
    # Fill between for drawdown
    ax2.fill_between(top_df['timestamp'], top_df['drawdown'], 0, color='red', alpha=0.3)
    ax2.plot(top_df['timestamp'], top_df['drawdown'], color='red', linewidth=1)
    ax2.set_ylabel("Drawdown (bps)", fontsize=12)
    ax2.set_xlabel("Time (or Trade Index)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{title_prefix.replace(' ', '_').replace('/', '-')}_equity_curve.png", dpi=300)
    plt.close()
    
    # Plot 2: Distribution of Realized Returns
    plt.figure(figsize=(10, 6))
    
    # Separate winners and losers for colored histogram
    bins = np.linspace(top_df['net_bps'].min(), top_df['net_bps'].max(), 50)
    
    winners = top_df[top_df['net_bps'] > 0]['net_bps']
    losers = top_df[top_df['net_bps'] <= 0]['net_bps']
    
    plt.hist(winners, bins=bins, color='green', alpha=0.6, label='Winners', edgecolor='black', linewidth=0.5)
    plt.hist(losers, bins=bins, color='red', alpha=0.6, label='Losers', edgecolor='black', linewidth=0.5)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=top_df['net_bps'].mean(), color='blue', linestyle='-.', linewidth=2, label=f'Mean: {top_df["net_bps"].mean():.2f} bps')
    
    plt.title(f"{title_prefix} - Distribution of Net Trade PnL", fontsize=14, fontweight='bold')
    plt.xlabel("Net PnL (bps)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{title_prefix.replace(' ', '_').replace('/', '-')}_distribution.png", dpi=300)
    plt.close()

def main():
    base_dir = Path("reports")
    out_dir = base_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = ["exp027", "exp029"]
    
    for exp in experiments:
        logger.info(f"Processing {exp}...")
        df = load_and_prep_data(str(base_dir / exp))
        if df is None:
            continue
            
        assets = df['asset'].unique()
        models = df['model'].unique()
        
        # Score the combinations
        scores = []
        for a in assets:
            for m in models:
                top = process_top_decile(df, a, m)
                if top is not None and len(top) > 50:
                    tot_pnl = top['cumulative_net_pnl'].iloc[-1]
                    sharpe = top['net_bps'].mean() / top['net_bps'].std() if top['net_bps'].std() > 0 else 0
                    scores.append((a, m, tot_pnl, sharpe, top))
                    
        # Sort by total PnL
        scores.sort(key=lambda x: x[2], reverse=True)
        
        # Take the top 3 best combinations for this experiment to visualize
        for a, m, pnl, sharpe, top_df in scores[:3]:
            logger.info(f"Visualizing {exp}: {a} x {m} (Total PnL: {pnl:.1f} bps, Sharpe: {sharpe:.3f})")
            title = f"{exp} {a} {m}"
            create_visualizations(top_df, title, out_dir)

if __name__ == "__main__":
    main()
