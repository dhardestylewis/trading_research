import pandas as pd
import logging

logger = logging.getLogger("replay_taker_markout")

def compute_net_taker_markout(
    gross_markout_bps: pd.Series,
    spread_bps: pd.Series,
    fee_bps: float = 4.0
) -> pd.Series:
    """
    Compute net executable edge given gross edge, assuming immediate aggressive execution.
    net_edge = gross_markout - spread_crossing - fees
    """
    # Assuming spread is quoted full spread, crossing costs half-spread entering and half-spread exiting (or similar logic).
    # For a round trip: total spread crossing = spread_bps (assuming spread remains constant).
    # We also apply a taker fee on entry and exit: 2 * fee_bps.
    net_edge = gross_markout_bps - spread_bps - (2 * fee_bps)
    return net_edge

def compute_latency_conditioned_net_markout(
    gross_markout_1s: pd.Series,
    latency_ms: int
) -> pd.Series:
    """
    Simulate signal decay factor based on latency.
    """
    logger.info(f"Computing net markout conditioned on {latency_ms}ms latency")
    decay_factor = min(latency_ms / 1000.0, 1.0)
    return gross_markout_1s * (1 - decay_factor)

def evaluate_execution_fidelity(df: pd.DataFrame, target="markout_1s", pred="pred_1s_gross"):
    """
    Generates the comparison between predicted gross edge vs realized net edge 
    across decile/ventile slices.
    """
    logger.info("Evaluating execution fidelity across score buckets...")
    if df.empty or pred not in df.columns or target not in df.columns:
        return pd.DataFrame()
        
    # Rank into 10 buckets
    try:
        df['score_decile'] = pd.qcut(df[pred], 10, labels=False, duplicates='drop')
    except ValueError:
        return pd.DataFrame()
    
    res = df.groupby('score_decile').agg({
        target: 'mean',
        pred: 'mean',
        'spread_bps': 'mean'
    })
    
    res['net_exec'] = compute_net_taker_markout(res[target], res['spread_bps'], fee_bps=4.0)
    return res
