import pandas as pd
import numpy as np
from src.models.foundation_challenger_exp026 import FoundationChallengerExp026

# Generate 500 rows, 150 features
X = pd.DataFrame(np.random.randn(500, 150), columns=[f"feat_{i}" for i in range(150)])
y = pd.DataFrame({"gross_move_bps_2": np.random.randn(500)})

challenger = FoundationChallengerExp026(horizons=[2])
try:
    challenger.fit(X, y)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
