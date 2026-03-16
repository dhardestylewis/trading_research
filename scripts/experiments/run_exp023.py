import os
import yaml
import logging
import pandas as pd
from pathlib import Path

from src.data.processing.build_1s_flow_state import build_offline_features
from src.models.flow_markout_model import GrossMarkoutModel, TrivialBaselines
from src.execution.replay_taker_markout import evaluate_execution_fidelity
from src.eval.freshness_study import evaluate_refresh_cadence
from src.reporting.exp023_report import generate_all_reports

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("exp023_runner")

def load_config(config_path="configs/experiments/crypto_microstructure_exp023.yaml"):
    logger.info(f"Loading config from {config_path}")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_branch_a_data(config):
    logger.info("--- Running Branch A: 90-day backfill and synchronization ---")
    data_dir = Path("data/processed/flow_bars")
    dfs = []
    
    universe = config.get('universe', {}).get('assets', ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    for asset in universe:
        sym = asset.replace("-", "").upper()
        file_path = data_dir / f"{sym}_1S_flow.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df['asset'] = sym
            dfs.append(df)
            
    if not dfs:
        logger.warning("No data loaded. Assuming scaffold mode.")
        return pd.DataFrame()
        
    raw_panel = pd.concat(dfs).sort_index()
    return build_offline_features(raw_panel)

def run_branch_b_modeling(config, panel: pd.DataFrame):
    logger.info("--- Running Branch B: Replay gross-signal replication ---")
    if panel.empty:
        return None, pd.DataFrame()
        
    model = GrossMarkoutModel()
    
    # Simple temporal split for offline backtest (80/20)
    train_size = int(len(panel) * 0.8)
    train_df = panel.iloc[:train_size]
    test_df = panel.iloc[train_size:].copy()
    
    model.train(train_df, target_1s="markout_1s", target_5s="markout_5s")
    
    preds = model.predict(test_df)
    test_df = test_df.join(preds)
    
    return model, test_df

def run_branch_c_execution(config, test_df: pd.DataFrame):
    logger.info("--- Running Branch C: Execution-conditioned replay ---")
    if test_df.empty:
        return pd.DataFrame()
        
    net_eval = evaluate_execution_fidelity(test_df, target="markout_1s", pred="pred_1s_gross")
    logger.info(f"Execution Fidelity Summary:\\n{net_eval}")
    return net_eval

def run_branch_d_freshness(config):
    logger.info("--- Running Branch D: Freshness and retraining study ---")
    train_windows = config['evaluation']['trailing_windows']
    cadences = config['evaluation']['refresh_cadences']
    
    res = evaluate_refresh_cadence(train_windows, cadences)
    logger.info(f"Freshness Study Results:\\n{res}")
    return res

def run_branch_e_live(config, champion_model, test_df):
    logger.info("--- Running Branch E: Live online paper harness ---")
    if champion_model is None or test_df.empty:
        return
        
    from src.execution.live_paper_taker import LivePaperTaker
    taker = LivePaperTaker(champion_model)
    
    # Simulate processing last 100 bars as live events
    sample = test_df.tail(100)
    for idx, row in sample.iterrows():
        features = row.to_dict()
        taker.process_1s_bar(idx, row['asset'], features, {'bid': 100, 'ask': 101})

def run_reporting(config, test_df, net_eval, freshness_res):
    logger.info("--- Running Reporting ---")
    generate_all_reports(test_df, net_eval, freshness_res)

def run_full_pipeline():
    logger.info("Starting exp023 pipeline: Live Execution-Conditioned Short-Horizon Flow Validation")
    try:
        config = load_config()
        panel = run_branch_a_data(config)
        model, test_df = run_branch_b_modeling(config, panel)
        net_eval = run_branch_c_execution(config, test_df)
        freshness_res = run_branch_d_freshness(config)
        run_branch_e_live(config, model, test_df)
        run_reporting(config, test_df, net_eval, freshness_res)
        logger.info("exp023 pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_full_pipeline()
