# %% [markdown]
"""
# Risk Management & Position Sizing

Volatility‐scaled sizing and drawdown simulations.
"""

# %% [code]
import pandas as pd
import numpy as np

# %% [code]
def position_size(equity: float, atr: pd.Series, risk_per_trade: float=0.01):
    dollar_risk = equity * risk_per_trade
    return dollar_risk / atr.iloc[-1]

# simulate sizing over time...
