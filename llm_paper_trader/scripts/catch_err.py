import subprocess
res = subprocess.run(["python", "jobs/02_price_and_trade.py"], capture_output=True, text=True)
with open("error.log", "w", encoding="utf-8") as f:
    f.write(res.stderr)
