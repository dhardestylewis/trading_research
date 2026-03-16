import pandas as pd
from pathlib import Path

csv_path = Path('reports/spike_tsfm/lane_discovery_results.csv')
rich_path = Path('data/processed/lane_discovery/panel_rich.parquet')

df = pd.read_csv(csv_path)
if 'timestamp' in df.columns:
    print("Timestamp already exists")
    exit(0)

rich = pd.read_parquet(rich_path)
all_timestamps = []

for asset in sorted(df['asset'].unique()):
    adf = rich[rich['asset']==asset].copy().sort_values('timestamp').reset_index(drop=True)
    valid = adf[adf['fwd_ret_8'].notna()].reset_index(drop=True)
    test_indices = list(range(300, len(valid), 55))
    
    asset_df = df[df['asset']==asset]
    assert len(test_indices) == len(asset_df), f"Len mismatch for {asset}: {len(test_indices)} vs {len(asset_df)}"
    
    ts = valid.iloc[test_indices]['timestamp'].values
    all_timestamps.extend(ts)

df['timestamp'] = all_timestamps
df.to_csv('reports/spike_tsfm/lane_discovery_results_with_time.csv', index=False)
print("Saved lane_discovery_results_with_time.csv successfully.")
