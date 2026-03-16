import pandas as pd
import numpy as np
import logging
import time

logger = logging.getLogger("build_1s_flow_state")

class FlowStateBuilder:
    """
    Folds raw tick/trade events into a 1-second rolling flow state matrix.
    Tracks signed volume, flow imbalance, trade count burst, VWAP, spreads.
    """
    def __init__(self, assets: list):
        self.assets = [a.replace('-', '').upper() for a in assets]
        self.state = {a: self._init_state() for a in self.assets}
        self.last_emit_time = time.time()

    def _init_state(self):
        return {
            'signed_volume_1s': 0.0,
            'buyer_maker_vol': 0.0,
            'seller_maker_vol': 0.0,
            'trade_count': 0,
            'vwap_num': 0.0,
            'vwap_den': 0.0,
            'best_bid': None,
            'best_ask': None,
            'bid_size': 0.0,
            'ask_size': 0.0
        }

    def process_tick(self, tick: dict) -> dict:
        """Update the rolling state with a new tick event."""
        asset = tick.get('s', '').upper()
        if asset not in self.state:
            return None

        event_type = tick.get('e')
        st = self.state[asset]
        
        if event_type == 'aggTrade':
            px = float(tick['p'])
            qty = float(tick['q'])
            is_buyer_maker = tick['m']
            
            st['trade_count'] += 1
            st['vwap_num'] += px * qty
            st['vwap_den'] += qty
            
            if is_buyer_maker: # Seller aggressor
                st['signed_volume_1s'] -= qty
                st['buyer_maker_vol'] += qty
            else: # Buyer aggressor
                st['signed_volume_1s'] += qty
                st['seller_maker_vol'] += qty
                
        elif 'b' in tick and 'a' in tick: # bookTicker
            st['best_bid'] = float(tick['b'])
            st['best_ask'] = float(tick['a'])
            st['bid_size'] = float(tick['B'])
            st['ask_size'] = float(tick['A'])
            
        current_time = time.time()
        if current_time - self.last_emit_time >= 1.0:
            features = self.emit_1s_bar()
            self.last_emit_time = current_time
            # Reset additive state
            for a in self.assets:
                s = self.state[a]
                s['signed_volume_1s'] = 0.0
                s['buyer_maker_vol'] = 0.0
                s['seller_maker_vol'] = 0.0
                s['trade_count'] = 0
                s['vwap_num'] = 0.0
                s['vwap_den'] = 0.0
            return features
        return None

    def emit_1s_bar(self) -> dict:
        """Produce the consolidated 1s feature row for ML scoring."""
        features = {}
        for asset, st in self.state.items():
            vwap = st['vwap_num'] / st['vwap_den'] if st['vwap_den'] > 0 else np.nan
            spread = (st['best_ask'] - st['best_bid']) if st['best_ask'] and st['best_bid'] else np.nan
            
            features[asset] = {
                'signed_volume_1s': st['signed_volume_1s'],
                'buyer_maker_seller_maker_imbalance': st['buyer_maker_vol'] - st['seller_maker_vol'],
                'trade_count_burst_intensity': st['trade_count'],
                'vwap_1s': vwap,
                'vwap_dislocation': (vwap / ((st['best_ask'] + st['best_bid'])/2) - 1) if (not np.isnan(vwap) and spread and spread > 0) else 0.0,
                'spread_bps': (spread / st['best_bid']) * 10000 if spread and st['best_bid'] else 0.0,
                'top_of_book_size': st['bid_size'] + st['ask_size'],
                'book_imbalance': (st['bid_size'] - st['ask_size']) / (st['bid_size'] + st['ask_size']) if (st['bid_size'] + st['ask_size']) > 0 else 0.0,
                'current_mid': (st['best_ask'] + st['best_bid'])/2 if st['best_ask'] and st['best_bid'] else np.nan,
                'best_bid': st['best_bid'],
                'best_ask': st['best_ask']
            }
        return features

def build_offline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Offline vectorized construction of the same 1s features for backfill."""
    logger.info("Building offline 1s flow state from raw backfill data...")
    if df.empty:
        return df
        
    res = df.copy()
    
    # Mock book features if missing (tick data primarily flow)
    if 'spread_bps' not in res.columns:
        res['spread_bps'] = 1.0 # 1 bps fallback
    if 'book_imbalance' not in res.columns:
        res['book_imbalance'] = 0.0
    if 'vwap_dislocation' not in res.columns and 'vwap' in res.columns and 'price' in res.columns:
        res['vwap_dislocation'] = (res['vwap'] / res['price']) - 1.0
        
    if 'seller_maker_vol' in res.columns and 'buyer_maker_vol' in res.columns:
        res['signed_volume_1s'] = res['seller_maker_vol'] - res['buyer_maker_vol']
        res['buyer_maker_seller_maker_imbalance'] = res['buyer_maker_vol'] - res['seller_maker_vol']
        
    if 'trade_count' in res.columns:
        res['trade_count_burst_intensity'] = res['trade_count']
        
    return res
