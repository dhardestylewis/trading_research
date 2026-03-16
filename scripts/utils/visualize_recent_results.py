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

def load_and_prep_data(experiment_dir: str, panel_df: pd.DataFrame):
    csv_path = Path(experiment_dir) / "partial_results_log.csv"
    if not csv_path.exists():
        logger.warning(f"File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    if df['timestamp'].dtype in ['int64', 'float64']:
        # Map integer indices to actual datetime from the panel
        logger.info("Mapping integer indices to datetimes from base panel...")
        df['idx_int'] = df['timestamp'].astype(int)
        
        # The base panel's index is the integer index matching these timestamps
        # Create a mapping series
        date_map = panel_df['timestamp']
        
        # Map it
        df['timestamp'] = df['idx_int'].map(date_map)
        
        # Drop rows where mapping failed (if any) and drop helper column
        missing = df['timestamp'].isna().sum()
        if missing > 0:
            logger.warning(f"Dropped {missing} rows due to unmappable timestamp indices.")
            df = df.dropna(subset=['timestamp'])
        df = df.drop(columns=['idx_int'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
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

def create_visualizations(top_df: pd.DataFrame, title_prefix: str, output_dir: Path, cost_bps: int = 14):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Plot 1: Cumulative PnL & Drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(top_df['timestamp'], top_df['cumulative_net_pnl'], color='blue', linewidth=2)
    ax1.set_title(f"{title_prefix} - Cumulative Net PnL (Top Decile, {cost_bps}bps cost)", fontsize=14, fontweight='bold')
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

def create_portfolio_visualizations(top_dfs: list, output_dir: Path):
    """Aggregate multiple DataFrames into a single portfolio timeline."""
    logger.info(f"Aggregating {len(top_dfs)} strategies into a combined portfolio...")
    
    # Concatenate all trades
    all_trades = pd.concat(top_dfs, ignore_index=True)
    all_trades = all_trades.sort_values('timestamp')
    
    # Group by timestamp to sum net PnL from trades happening at the same time
    portfolio = all_trades.groupby('timestamp').agg(
        net_bps=('net_bps', 'sum'),
        trade_count=('net_bps', 'count')
    ).reset_index()
    
    # Sort and calc cumulative
    portfolio = portfolio.sort_values('timestamp')
    portfolio['cumulative_net_pnl'] = portfolio['net_bps'].cumsum()
    
    # Calculate drawdown
    running_max = portfolio['cumulative_net_pnl'].cummax()
    portfolio['drawdown'] = portfolio['cumulative_net_pnl'] - running_max
    
    # Plot Portfolio Equity Curve & Drawdown
    sns.set_theme(style="whitegrid", palette="muted")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(portfolio['timestamp'], portfolio['cumulative_net_pnl'], color='purple', linewidth=2)
    ax1.set_title(f"Combined Portfolio - Cumulative Net PnL (Simultaneous Trading, Base/Adjusted Costs)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative PnL (bps)", fontsize=12)
    
    ax2.fill_between(portfolio['timestamp'], portfolio['drawdown'], 0, color='red', alpha=0.3)
    ax2.plot(portfolio['timestamp'], portfolio['drawdown'], color='red', linewidth=1)
    ax2.set_ylabel("Drawdown (bps)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_portfolio_equity_curve.png", dpi=300)
    plt.close()
    
    # Plot Portfolio Distribution
    plt.figure(figsize=(10, 6))
    bins = np.linspace(portfolio['net_bps'].min(), portfolio['net_bps'].max(), 50)
    
    winners = portfolio[portfolio['net_bps'] > 0]['net_bps']
    losers = portfolio[portfolio['net_bps'] <= 0]['net_bps']
    
    plt.hist(winners, bins=bins, color='green', alpha=0.6, label='Winners', edgecolor='black', linewidth=0.5)
    plt.hist(losers, bins=bins, color='red', alpha=0.6, label='Losers', edgecolor='black', linewidth=0.5)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    mean_bps = portfolio["net_bps"].mean()
    plt.axvline(x=mean_bps, color='blue', linestyle='-.', linewidth=2, label=f'Mean: {mean_bps:.2f} bps/period')
    
    plt.title("Combined Portfolio - Distribution of Net Period PnL", fontsize=14, fontweight='bold')
    plt.xlabel("Net PnL (bps)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_portfolio_distribution.png", dpi=300)
    plt.close()
    
    sharpe = mean_bps / portfolio['net_bps'].std() if portfolio['net_bps'].std() > 0 else 0
    total = portfolio['cumulative_net_pnl'].iloc[-1]
    logger.info(f"Portfolio Total PnL: {total:.1f} bps | Sharpe per period: {sharpe:.3f} | Total periods traded: {len(portfolio)}")


def main():
    base_dir = Path("reports")
    out_dir = base_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base panel once to map dates
    panel_path = Path("data/processed/panel_expanded/panel.parquet")
    logger.info(f"Loading base panel from {panel_path} to map dates...")
    panel_df = pd.read_parquet(panel_path)
    
    all_top_dfs = []
    
    # Target 1: exp027 (APT-USD XGB, SOL-USD TabPFN)
    logger.info("Processing exp027 targets...")
    df_027 = load_and_prep_data(str(base_dir / 'exp027'), panel_df)
    if df_027 is not None:
        targets_027 = [('APT-USD', 'XGBoostCounterBaseline'), ('SOL-USD', 'TabPFNTopDecile')]
        for a, m in targets_027:
            top = process_top_decile(df_027, a, m)
            if top is not None:
                title = f"exp027 {a} {m}"
                create_visualizations(top, title, out_dir, COST_BPS)
                all_top_dfs.append(top)
                
    # Target 2: exp029 (SOL-USD TabPFN, SUI-USD TabPFN)
    logger.info("Processing exp029 targets...")
    df_029 = load_and_prep_data(str(base_dir / 'exp029'), panel_df)
    if df_029 is not None:
        targets_029 = [('SOL-USD', 'TabPFNTopDecile'), ('SUI-USD', 'TabPFNTopDecile')]
        for a, m in targets_029:
            top = process_top_decile(df_029, a, m)
            if top is not None:
                title = f"exp029 {a} {m}"
                create_visualizations(top, title, out_dir, COST_BPS)
                all_top_dfs.append(top)
                
    # Target 3: Spikes (EWJ, FXA, FXE, INDA, UVXY, XLE, XLF, XLU)
    logger.info("Processing spike targets...")
    spike_path = base_dir / "spike_tsfm" / "lane_discovery_results_with_time.csv"
    if spike_path.exists():
        spike_df = pd.read_csv(spike_path)
        spike_df['timestamp'] = pd.to_datetime(spike_df['timestamp'], utc=True)
        # Filters to only 'fires' and specific targets
        target_spikes = ['EWJ', 'FXA', 'FXE', 'INDA', 'UVXY', 'XLE', 'XLF', 'XLU']
        spike_cost_bps = 5 # spike script uses 5 bps
        for a in target_spikes:
            sf = spike_df[(spike_df['asset'] == a) & (spike_df['fires'])].copy()
            if not sf.empty:
                sf = sf.sort_values('timestamp')
                # For spike, PnL is already calculated with cost
                sf['net_bps'] = sf['pnl']
                sf['cumulative_net_pnl'] = sf['net_bps'].cumsum()
                running_max = sf['cumulative_net_pnl'].cummax()
                sf['drawdown'] = sf['cumulative_net_pnl'] - running_max
                
                title = f"spike_lane_discovery {a}"
                create_visualizations(sf, title, out_dir, spike_cost_bps)
                all_top_dfs.append(sf)

    # Visualize combined portfolio
    if all_top_dfs:
        create_portfolio_visualizations(all_top_dfs, out_dir)

if __name__ == "__main__":
    main()
