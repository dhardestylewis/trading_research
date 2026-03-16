import pandas as pd
df = pd.read_parquet('data/processed/panel_expanded/panel.parquet')
dates = sorted(df['timestamp'].unique())
print(f'Total dates: {len(dates)}')
print(f'1000th date: {dates[1000]}')
if len(dates) > 48142:
    print(f'48142th date (from exp): {dates[48142]}')
else:
    print(f'Index 48142 is out of bounds for {len(dates)} dates.')
