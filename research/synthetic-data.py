# %% [markdown]
"""
# Synthetic Data & Stress Testing

Generate GBM and GARCH series to stress‐test agents.
"""

# %% [code]
import numpy as np
import pandas as pd

# %% [code]
def generate_gbm(n: int=1000, mu: float=0.0002, sigma: float=0.01):
    dt = 1/252
    returns = np.random.normal(mu*dt, sigma*np.sqrt(dt), n)
    price = 100 * np.exp(np.cumsum(returns))
    return pd.Series(price)

# %% [code]
gbm = generate_gbm()
