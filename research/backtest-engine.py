# %% [markdown]
"""
# Vectorized Backtesting Engine

Define rules, compute P&L, and plot equity curves & drawdowns.
"""

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [code]
class VectorBacktester:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.positions = pd.Series(0, index=self.df.index)
        self.returns = self.df["close"].pct_change().fillna(0)

    def apply_signal(self, signals: pd.Series):
        self.positions = signals.shift().fillna(0)
        pnl = self.positions * self.returns
        self.df["equity_curve"] = (1 + pnl).cumprod()
        return self.df

# %% [code]
# Example usage
# signals = pd.Series([...], index=df.index)
# result = VectorBacktester(df).apply_signal(signals)
# plt.plot(result["equity_curve"])
