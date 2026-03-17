#!/bin/bash
set -e

# Change to the script's directory
cd "$(dirname "$0")"

echo "=== LLM PAPER TRADER PIPELINE ==="
echo "Started at: $(date)"

# 1. Update active markets
python jobs/01_scan_markets.py

# 2. Price and execute trades
python jobs/02_price_and_trade.py

# 3. Grade portfolio Mark-to-Market
python jobs/03_grade_pnl.py

echo "=== PIPELINE COMPLETE ==="
