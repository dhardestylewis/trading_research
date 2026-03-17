import sqlite3
import os
from datetime import datetime

class DB:
    def __init__(self, db_path: str = "db.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Markets we track
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS markets (
                    market_id TEXT PRIMARY KEY,
                    title TEXT,
                    resolution_date DATETIME,
                    volume REAL,
                    question TEXT,
                    exchange TEXT DEFAULT 'POLYMARKET',
                    status TEXT DEFAULT 'OPEN'
                )
            ''')
            
            # LLM Predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    market_id TEXT,
                    p_llm REAL,
                    context_used TEXT,
                    reasoning TEXT,
                    FOREIGN KEY (market_id) REFERENCES markets(market_id)
                )
            ''')
            
            # Paper Trades
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    market_id TEXT,
                    side TEXT,
                    fill_price REAL,
                    size_shares REAL,
                    status TEXT DEFAULT 'OPEN',
                    pnl REAL DEFAULT 0.0,
                    FOREIGN KEY (market_id) REFERENCES markets(market_id)
                )
            ''')
            
            # Trading Rules (Memory)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_text TEXT,
                    domain TEXT,
                    weight REAL,
                    created_at DATETIME
                )
            ''')
            
            # Reflection Logs (State)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reflection_logs (
                    trade_id INTEGER PRIMARY KEY,
                    pnl_at_reflection REAL,
                    reflected_at DATETIME,
                    FOREIGN KEY (trade_id) REFERENCES paper_trades(id)
                )
            ''')
            
            conn.commit()

    def save_market(self, market_id: str, title: str, resolution_date: str, volume: float, question: str, exchange: str = 'POLYMARKET'):
        with self._get_conn() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO markets (market_id, title, resolution_date, volume, question, exchange)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (market_id, title, resolution_date, volume, question, exchange))
            conn.commit()

    def get_open_markets(self):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM markets WHERE status = 'OPEN'")
            return [dict(row) for row in cursor.fetchall()]

    def save_prediction(self, market_id: str, p_llm: float, context_used: str, reasoning: str):
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO predictions (timestamp, market_id, p_llm, context_used, reasoning)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.utcnow().isoformat(), market_id, p_llm, context_used, reasoning))
            conn.commit()
            
    def get_predictions_for_market(self, market_id: str):
         with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM predictions WHERE market_id = ? ORDER BY timestamp DESC", (market_id,))
            return [dict(row) for row in cursor.fetchall()]

    def save_trade(self, market_id: str, side: str, fill_price: float, size_shares: float):
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO paper_trades (timestamp, market_id, side, fill_price, size_shares)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.utcnow().isoformat(), market_id, side, fill_price, size_shares))
            conn.commit()
            
    def get_open_trades(self):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM paper_trades WHERE status = 'OPEN'")
            return [dict(row) for row in cursor.fetchall()]
            
    def update_trade_pnl(self, trade_id: int, status: str, pnl: float):
         with self._get_conn() as conn:
            conn.execute('''
                UPDATE paper_trades
                SET status = ?, pnl = ?
                WHERE id = ?
            ''', (status, pnl, trade_id))
            conn.commit()

    def save_trading_rule(self, rule_text: str, domain: str, weight: float):
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO trading_rules (rule_text, domain, weight, created_at)
                VALUES (?, ?, ?, ?)
            ''', (rule_text, domain, weight, datetime.utcnow().isoformat()))
            conn.commit()

    def get_top_trading_rules(self, limit: int = 10):
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM trading_rules ORDER BY weight DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def log_reflection(self, trade_id: int, pnl_at_reflection: float):
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO reflection_logs (trade_id, pnl_at_reflection, reflected_at)
                VALUES (?, ?, ?)
            ''', (trade_id, pnl_at_reflection, datetime.utcnow().isoformat()))
            conn.commit()

    def get_unreflected_losing_trades(self, loss_threshold: float = -10.0, limit: int = 5):
        # We need to join paper_trades with markets to get context, and filter out already reflected trades
        with self._get_conn() as conn:
            cursor = conn.execute('''
                SELECT pt.*, m.question, m.title, m.exchange 
                FROM paper_trades pt
                JOIN markets m ON pt.market_id = m.market_id
                LEFT JOIN reflection_logs rl ON pt.id = rl.trade_id
                WHERE rl.trade_id IS NULL AND pt.pnl <= ?
                ORDER BY pt.pnl ASC
                LIMIT ?
            ''', (loss_threshold, limit))
            return [dict(row) for row in cursor.fetchall()]
            
    def apply_temporal_decay(self, decay_factor: float = 0.9, prune_threshold: float = 2.0):
        with self._get_conn() as conn:
            # Decay all weights
            conn.execute("UPDATE trading_rules SET weight = weight * ?", (decay_factor,))
            # Delete weak rules
            conn.execute("DELETE FROM trading_rules WHERE weight < ?", (prune_threshold,))
            conn.commit()

