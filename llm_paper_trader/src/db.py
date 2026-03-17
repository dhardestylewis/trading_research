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
