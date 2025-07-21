import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, rf: float=0.0, periods_per_year: int=252) -> float:
    excess = returns - rf/periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()

def sortino_ratio(returns: pd.Series, rf: float=0.0, periods_per_year: int=252) -> float:
    excess = returns - rf/periods_per_year
    neg_std = returns[returns<0].std()
    return np.sqrt(periods_per_year) * excess.mean() / neg_std

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()

def cagr(equity: pd.Series) -> float:
    n = len(equity)
    return (equity.iloc[-1])**(252/n) - 1
