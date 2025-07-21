# %% [markdown]
"""
# Walk‐Forward & Robustness Testing

Rolling-window in‐sample/out‐of‐sample splits and performance heatmap.
"""

# %% [code]
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from research.backtest_engine import VectorBacktester

# %% [code]
def walk_forward(df, in_s=252, out_s=63):
    results = {}
    for start in range(0, len(df)-in_s-out_s, out_s):
        train = df.iloc[start:start+in_s]
        test = df.iloc[start+in_s:start+in_s+out_s]
        # fit parameters on train...
        # evaluate on test...
        results[f"{start}"] = {"train_sharpe":..., "test_sharpe":...}
    return pd.DataFrame(results).T

wf = walk_forward(df)
sns.heatmap(wf.astype(float), annot=True)
