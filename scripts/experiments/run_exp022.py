import os
import yaml
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from src.reporting.exp022_report import generate_report

log = logging.getLogger("run_exp022")
logging.basicConfig(level=logging.INFO)

def load_config(path="configs/experiments/crypto_flow_exp022.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(config):
    data_dir = Path(config['dataset']['data_dir'])
    freq = config['dataset']['frequency']
    universe = config['dataset']['universe']
    
    dfs = []
    for sym in universe:
        sym_upper = sym.replace("-", "").upper()
        file_path = data_dir / f"{sym_upper}_{freq}_flow.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df['asset'] = sym
            dfs.append(df)
        else:
            log.warning(f"File not found: {file_path}")
            
    if not dfs:
        raise ValueError("No data loaded. Check build_flow_bars.py execution.")
        
    panel = pd.concat(dfs)
    
    # Optional: filter by date
    start_date = config['dataset'].get('start_date')
    end_date = config['dataset'].get('end_date')
    
    if start_date:
        panel = panel[panel.index >= pd.to_datetime(start_date)]
    if end_date:
        panel = panel[panel.index <= pd.to_datetime(end_date)]
        
    return panel.sort_index()

def build_features(df, config):
    """Generate lagged flow features based on config."""
    log.info("Building lagged flow features...")
    features = []
    
    lags = config['features'].get('lags', [1, 5, 10, 30])
    
    for asset, group in df.groupby('asset'):
        # Sort to ensure proper lagging
        g = group.copy().sort_index()
        
        if config['features'].get('flow_imbalance', True):
            for l in lags:
                g[f'flow_imb_lag_{l}'] = g['flow_imbalance'].shift(l)
                
        if config['features'].get('trade_intensity', True):
            for l in lags:
                g[f'trade_count_lag_{l}'] = g['trade_count'].shift(l)
                
        if config['features'].get('vwap_dislocation', True):
            for l in lags:
                # VWAP vs Close price difference
                g[f'vwap_disloc_lag_{l}'] = (g['vwap'].shift(l) / g['price'].shift(l)) - 1
                
        if config['features'].get('signed_volume_shock', True):
            for l in lags:
                signed_vol = g['seller_maker_vol'] - g['buyer_maker_vol']
                g[f'signed_vol_lag_{l}'] = signed_vol.shift(l)
                
        features.append(g)
        
    panel = pd.concat(features).sort_index()
    panel.dropna(inplace=True) # drop NA from lagging logic
    return panel

def train_and_evaluate(panel, target, config):
    """Walk-forward train and evaluate for a single target."""
    log.info(f"--- Training target: {target} ---")
    
    # Feature columns (everything that's a lag)
    feature_cols = [c for c in panel.columns if '_lag_' in c]
    
    if not feature_cols:
        raise ValueError("No features generated.")
        
    dates = np.unique(panel.index.date)
    train_days = config['model']['validation']['train_size_days']
    test_days = config['model']['validation']['test_size_days']
    step_days = config['model']['validation']['step_size_days']
    
    predictions = []
    feature_importances = []
    
    for start_idx in range(0, len(dates) - train_days - test_days, step_days):
        train_start = dates[start_idx]
        train_end = dates[start_idx + train_days - 1]
        test_start = dates[start_idx + train_days]
        test_end = dates[start_idx + train_days + test_days - 1]
        
        train_mask = (panel.index.date >= train_start) & (panel.index.date <= train_end)
        test_mask = (panel.index.date >= test_start) & (panel.index.date <= test_end)
        
        train_df = panel[train_mask]
        test_df = panel[test_mask]
        
        if len(train_df) < 100 or len(test_df) < 10:
            continue
            
        X_train, y_train = train_df[feature_cols], train_df[target]
        X_test, y_test = test_df[feature_cols], test_df[target]
        
        # Train LightGBM
        params = config['model']['params']
        params['verbose'] = -1
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, dtrain, num_boost_round=params.get('n_estimators', 100))
        
        preds = model.predict(X_test)
        
        res = test_df[['asset', target]].copy()
        res['prediction'] = preds
        predictions.append(res)
        
        fi = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain')
        })
        feature_importances.append(fi)
        
    if not predictions:
        log.warning(f"No predictions generated for {target}. Check dataset length.")
        return {"spearman": 0, "sign_acc": 0, "feature_importance": pd.DataFrame()}
        
    all_preds = pd.concat(predictions)
    
    # Metrics
    spearman = all_preds.groupby('asset').apply(lambda x: x[target].corr(x['prediction'], method='spearman')).mean()
    
    # Sign accuracy (excluding zero predictions or zero targets if any)
    valid_signs = all_preds[(all_preds[target] != 0) & (all_preds['prediction'] != 0)]
    sign_acc = (np.sign(valid_signs[target]) == np.sign(valid_signs['prediction'])).mean() if len(valid_signs) > 0 else 0
    
    # Feature Importance Aggregation
    fi_df = pd.concat(feature_importances).groupby('feature').mean().reset_index()
    fi_df.sort_values('importance', ascending=False, inplace=True)
    
    log.info(f"{target} -> Spearman: {spearman:.4f} | Sign Acc: {sign_acc:.2%}")
    
    return {
        "spearman": spearman,
        "sign_acc": sign_acc,
        "feature_importance": fi_df
    }

def main():
    config = load_config()
    log.info(f"Loaded config: {config['experiment_name']}")
    
    try:
        panel = load_data(config)
    except ValueError as e:
        log.error(e)
        return
        
    log.info(f"Loaded {len(panel)} raw flow bars.")
    
    # Build features
    enhanced_panel = build_features(panel, config)
    log.info(f"Enhanced panel shape: {enhanced_panel.shape}")
    
    targets = config['targets']
    metrics = {}
    
    for target in targets:
        if target not in enhanced_panel.columns:
            log.warning(f"Target {target} not found in panel columns. Skipping.")
            continue
            
        metrics[target] = train_and_evaluate(enhanced_panel, target, config)
        
    # Generate report
    report_dir = Path("reports/exp022")
    generate_report(metrics, report_dir)

if __name__ == "__main__":
    main()
