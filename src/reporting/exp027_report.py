import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any

def generate_exp027_report(
    config: Dict[str, Any],
    scorecard: pd.DataFrame,
    report_dir: str
) -> str:
    """
    Generates a Markdown report isolating the Top Decile (Decile 9) economics 
    from the foundation models mapping (exp027).
    """
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir) / f"{config['experiment_name']}_report.md"
    
    friction_bps = config.get('execution_assumptions', {}).get('friction_bps', 14)
    min_trades = config.get('evaluation', {}).get('min_decile_9_trades', 100)
    required_net = config.get('evaluation', {}).get('required_decile_9_net_expectancy', 5.0)
    
    # Analyze the scorecard
    if not scorecard.empty:
        # Check constraints
        scorecard['meets_trade_req'] = scorecard['trade_count'] >= min_trades
        scorecard['meets_net_req'] = (scorecard['median_net_bps'] > required_net) & (scorecard['trimmed_mean_net_bps'] > required_net)
        scorecard['pass'] = scorecard['meets_trade_req'] & scorecard['meets_net_req']
        
        pass_count = scorecard['pass'].sum()
        total_cells = len(scorecard)
        verdict = "PASS" if pass_count > 0 else "FAIL (No models retained robust Decile 9 edge)"
    else:
        verdict = "FAIL (Empty Scorecard)"
    
    with open(report_path, 'w') as f:
        f.write(f"# {config['experiment_name']}: Foundation Top-Decile Edge\n\n")
        f.write("## 1. Objective and Setup\n")
        f.write("Isolate and robustly evaluate the top-decile predictive edge discovered by pre-trained foundation models (TabPFN) at the 8h horizon.\n\n")
        f.write("### Target Variables\n")
        f.write(f"- Friction hurdle: `{friction_bps}` bps round-trip\n")
        f.write(f"- Required Minimum Trades in Top Decile: `{min_trades}`\n")
        f.write(f"- Required Robust Net Expectancy > `{required_net}` bps\n\n")
        
        f.write("## 2. Hard Gates Verdict\n")
        f.write(f"- Cells tested: {len(scorecard) if not scorecard.empty else 0}\n")
        f.write(f"- Cells surviving Gate requirements: {pass_count if not scorecard.empty else 0}\n\n")
        f.write(f"**OVERALL BRANCH VERDICT: {verdict}**\n\n")
        
        f.write("## 3. Top Decile (Decile 9) Isolation Scorecard\n")
        f.write("The table below exclusively displays prediction metrics isolated for the 9th decile (top 10%) of model conviction, sorted by model and asset.\n\n")
        
        if not scorecard.empty:
            # Format numeric columns for clean markdown
            formatted_sc = scorecard.copy()
            for col in ['predicted_gross_bps', 'realized_gross_bps_mean', 'realized_net_bps_mean', 'median_net_bps', 'trimmed_mean_net_bps']:
                formatted_sc[col] = formatted_sc[col].round(2)
            
            md_table = formatted_sc.to_markdown(index=False)
            f.write(f"{md_table}\n\n")
        else:
            f.write("*No trades survived processing.*\n")
            
    return str(report_path)
