"""
Run script for exp029: Expanded Foundation Universe
Tests TabPFN top-decile edge across 4h, 8h, 12h horizons on all assets.
Reuses exp027 infrastructure with multi-horizon support.
"""
import argparse
import sys
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.build_foundation_features import FoundationFeatureBuilder
from src.models.foundation_edge_models_exp027 import get_challenger
from src.eval.top_decile_scorecard_exp027 import evaluate_top_decile
from src.reporting.exp027_report import generate_exp027_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run exp029: Expanded Foundation Universe")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Run a single fast fold")
    return parser.parse_args()

def run_pipeline():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Starting {config['experiment_name']}")
    
    # Load Data
    panel_path = Path(config["data"]["panel_path"])
    logger.info(f"Loading data from {panel_path}...")
    raw_df = pd.read_parquet(panel_path)
    
    # Walk-forward params
    train_days = config['walk_forward']['train_days']
    test_days = config['walk_forward']['test_days']
    embargo_days = config['walk_forward']['embargo_days']
    max_train_samples = config['walk_forward']['max_train_samples']
    symbols = config['data']['symbols']
    horizons = config['targets']['horizons']
    
    all_results = []
    
    # Build features for each horizon separately
    for horizon in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing horizon: {horizon}h")
        logger.info(f"{'='*60}")
        
        # Build features for this horizon
        horizon_config = config.copy()
        horizon_config['targets'] = {'horizons': [horizon]}
        builder = FoundationFeatureBuilder(horizon_config)
        rich_df, feature_cols = builder.build_features(raw_df)
        
        target_col = f"fwd_ret_{horizon}"
        if target_col not in rich_df.columns:
            logger.warning(f"Target column '{target_col}' not found, skipping {horizon}h")
            continue
        
        start_time = rich_df['timestamp'].min()
        end_time = rich_df['timestamp'].max()
        
        if args.dry_run:
            end_time = start_time + pd.Timedelta(days=train_days + embargo_days + test_days)
        
        fold = 0
        current_time = start_time
        
        while current_time + pd.Timedelta(days=train_days + embargo_days + test_days) <= end_time:
            train_start = current_time
            train_end = train_start + pd.Timedelta(days=train_days)
            test_start = train_end + pd.Timedelta(days=embargo_days)
            test_end = test_start + pd.Timedelta(days=test_days)
            
            logger.info(f"--- {horizon}h Fold {fold} | {train_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} ---")
            
            for asset in symbols:
                asset_df = rich_df[rich_df['symbol'] == asset].copy() if 'symbol' in rich_df.columns else rich_df[rich_df['asset'] == asset].copy()
                if asset_df.empty:
                    continue
                
                train_slice = asset_df[(asset_df['timestamp'] >= train_start) & (asset_df['timestamp'] < train_end)].copy()
                test_slice = asset_df[(asset_df['timestamp'] >= test_start) & (asset_df['timestamp'] < test_end)].copy()
                
                if len(train_slice) < 100 or len(test_slice) < 24:
                    continue
                
                y_train = train_slice[target_col].dropna()
                if len(y_train) < 100:
                    continue
                
                train_valid = builder.enforce_n_constraint(train_slice.loc[y_train.index], max_n=max_train_samples)
                y_train = y_train.loc[train_valid.index]
                
                X_train_pca = builder.fit_transform_pca(train_valid[feature_cols])
                
                models_to_test = config['models']['challengers']
                for model_name in models_to_test:
                    model = get_challenger(model_name, config)
                    
                    try:
                        model.fit(X_train_pca, y_train * 10000)
                        
                        valid_test = test_slice[test_slice[target_col].notna()]
                        y_test = valid_test[target_col] * 10000
                        
                        X_test_pca = builder.transform_pca(valid_test[feature_cols])
                        preds = model.predict(X_test_pca)
                        
                        res_df = pd.DataFrame({
                            'timestamp': valid_test.index,
                            'asset': asset,
                            'horizon': horizon,
                            'model': model_name,
                            'predicted_move': preds,
                            'realized_move_bps': y_test.values
                        })
                        all_results.append(res_df)
                        
                        # Incremental save
                        out_dir = Path(f"reports/{config['experiment_name']}")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        temp_out = out_dir / "partial_results_log.csv"
                        res_df.to_csv(temp_out, mode='a', header=not temp_out.exists(), index=False)
                        
                    except Exception as e:
                        logger.error(f"Model {model_name} failed on {asset} {horizon}h: {str(e)}")
            
            fold += 1
            current_time = test_start
            
            if args.dry_run:
                break
    
    if not all_results:
        logger.error("No predictions generated!")
        sys.exit(1)
    
    full_results_df = pd.concat(all_results, ignore_index=True)
    
    scorecard = evaluate_top_decile(full_results_df, config)
    
    report_dir = f"reports/{config['experiment_name']}"
    report_path = generate_exp027_report(config, scorecard, report_dir)
    
    logger.info(f"\n{config['experiment_name']} complete. Report written to {report_path}.")

if __name__ == "__main__":
    run_pipeline()
