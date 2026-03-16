"""
Run script for exp027: Foundation Top-Decile Edge
"""
import argparse
import sys
import yaml
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

# Local imports
from src.data.build_foundation_features import FoundationFeatureBuilder
from src.models.foundation_edge_models_exp027 import get_challenger
from src.eval.top_decile_scorecard_exp027 import evaluate_top_decile
from src.reporting.exp027_report import generate_exp027_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run exp027: Foundation Top-Decile Edge")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Run a single fast fold for validation")
    return parser.parse_args()

def run_pipeline():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Starting {config['experiment_name']}")
    
    # 1. Load Data
    panel_path = Path(config["data"]["panel_path"])
    logger.info(f"Loading data from {panel_path}...")
    raw_df = pd.read_parquet(panel_path)
    
    # 2. Build Features (PCA bounded)
    builder = FoundationFeatureBuilder(config)
    rich_df, feature_cols = builder.build_features(raw_df)
    
    # 3. Standardize Walk-forward boundaries
    # Using simple chronological walk-forward.
    train_days = config['walk_forward']['train_days']
    test_days = config['walk_forward']['test_days']
    embargo_days = config['walk_forward']['embargo_days']
    max_train_samples = config['walk_forward']['max_train_samples']
    
    symbols = config['data']['symbols']
    all_results = []
    
    start_time = rich_df['timestamp'].min()
    end_time = rich_df['timestamp'].max()
    
    if args.dry_run:
        logger.info("DRY RUN: Testing a single slice")
        # Ensure we have enough data for 1 fit
        end_time = start_time + pd.Timedelta(days=train_days + embargo_days + test_days)

    fold = 0
    current_time = start_time
    
    logger.info(f"Targeting horizons: {config['targets']['horizons']}h exclusively...")
    
    while current_time + pd.Timedelta(days=train_days + embargo_days + test_days) <= end_time:
        train_start = current_time
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end + pd.Timedelta(days=embargo_days)
        test_end = test_start + pd.Timedelta(days=test_days)
        
        logger.info(f"\\n--- Fold {fold} | Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} | Test: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} ---")
        
        for asset in symbols:
            asset_df = rich_df[rich_df['symbol'] == asset].copy() if 'symbol' in rich_df.columns else rich_df[rich_df['asset'] == asset].copy()
            if asset_df.empty:
                continue
                
            train_slice = asset_df[(asset_df['timestamp'] >= train_start) & (asset_df['timestamp'] < train_end)].copy()
            test_slice = asset_df[(asset_df['timestamp'] >= test_start) & (asset_df['timestamp'] < test_end)].copy()
            
            if len(train_slice) < 100 or len(test_slice) < 24:
                continue
                
            # Filter NaNs for the target horizon
            y_col = "fwd_ret_8"
            y_train = train_slice[y_col].dropna()
            
            # Ensure we have enough data for robust PCA components
            if len(y_train) < 100:
                continue
                
            # Subsample train directly 
            train_valid = builder.enforce_n_constraint(train_slice.loc[y_train.index], max_n=max_train_samples)
            y_train = y_train.loc[train_valid.index]
            
            # Apply PCA
            X_train_pca = builder.fit_transform_pca(train_valid[feature_cols])
            
            # Iterate models
            models_to_test = config['models']['challengers']
            for model_name in models_to_test:
                logger.info(f"Fitting {model_name} for {asset}...")
                model = get_challenger(model_name, config)
                
                try: # Catch foundation limits just in case
                    model.fit(X_train_pca, y_train * 10000) # Target in bps
                    
                    # Test execution
                    valid_test = test_slice[test_slice[y_col].notna()]
                    y_test = valid_test[y_col] * 10000 
                    
                    X_test_pca = builder.transform_pca(valid_test[feature_cols])
                    preds = model.predict(X_test_pca)
                    
                    res_df = pd.DataFrame({
                        'timestamp': valid_test.index,
                        'asset': asset,
                        'horizon': 8,
                        'model': model_name,
                        'predicted_move': preds,
                        'realized_move_bps': y_test.values
                    })
                    all_results.append(res_df)
                    
                    # Intermediary save for real-time inspection
                    out_dir = Path(f"reports/{config['experiment_name']}")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    temp_out = out_dir / "partial_results_log.csv"
                    res_df.to_csv(temp_out, mode='a', header=not temp_out.exists(), index=False)
                    
                except Exception as e:
                    logger.error(f"Model {model_name} failed on {asset}: {str(e)}")
                    
        fold += 1
        current_time = test_start # Forward chaining without overlap
        
        if args.dry_run:
            break
            
    if not all_results:
        logger.error("No predictions generated!")
        sys.exit(1)
        
    full_results_df = pd.concat(all_results, ignore_index=True)
    
    # Evaluate exclusive Top Decile Isolation
    scorecard = evaluate_top_decile(full_results_df, config)
    
    # Generate Report
    report_dir = f"reports/{config['experiment_name']}"
    report_path = generate_exp027_report(config, scorecard, report_dir)
    
    logger.info(f"\n{config['experiment_name']} complete. Report written to {report_path}.")

if __name__ == "__main__":
    run_pipeline()
