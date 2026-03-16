import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def evaluate_top_decile(results_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Evaluates the economic performance explicitly filtering for Decile 9 (Top Decile).
    
    Inputs:
        results_df: Contains 'timestamp', 'asset', 'horizon', 'model', 'predicted_move', 'realized_move_bps'
    Returns:
        pd.DataFrame containing the isolated scorecard for only the top decile predictions.
    """
    friction_bps = config.get('execution_assumptions', {}).get('friction_bps', 14)
    logger.info(f"Evaluating top decile economics against {friction_bps} bps friction.")
    
    # 1. Calculate universal score buckets per model × horizon slice globally using overall test distribution
    def assign_deciles(group):
        try:
            return pd.qcut(group, 10, labels=False, duplicates='drop')
        except ValueError:
            logger.warning(f"Could not calculate quantiles for {group.name}, assigning 0.")
            return np.zeros(len(group))
            
    results_df['score_bucket'] = results_df.groupby(['model', 'horizon'])['predicted_move'].transform(assign_deciles)
    
    # 2. Isolate exclusively Decile 9
    top_decile_df = results_df[results_df['score_bucket'] == 9].copy()
    
    # 3. Calculate execution economics
    top_decile_df['net_move_bps'] = top_decile_df['realized_move_bps'] - friction_bps
    
    # 4. Aggregate strictly by asset × horizon(8h) × model
    agg_funcs = {
        'timestamp': 'count',
        'predicted_move': 'mean',
        'realized_move_bps': 'mean',
        'net_move_bps': ['mean', 'median', lambda x: x.clip(lower=x.quantile(0.05), upper=x.quantile(0.95)).mean()]
    }
    
    scorecard = top_decile_df.groupby(['asset', 'horizon', 'model']).agg(agg_funcs).reset_index()
    
    # Flatten multi-index columns
    scorecard.columns = [
        'asset', 'horizon', 'model', 
        'trade_count', 'predicted_gross_bps', 'realized_gross_bps_mean', 
        'realized_net_bps_mean', 'median_net_bps', 'trimmed_mean_net_bps'
    ]
    
    scorecard['estimated_cost_bps'] = friction_bps
    
    # Reorder
    cols = [
        'asset', 'horizon', 'model', 'trade_count', 
        'predicted_gross_bps', 'realized_gross_bps_mean', 'estimated_cost_bps',
        'realized_net_bps_mean', 'median_net_bps', 'trimmed_mean_net_bps'
    ]
    scorecard = scorecard[cols]
    
    return scorecard
