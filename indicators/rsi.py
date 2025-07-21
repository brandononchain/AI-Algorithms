import pandas as pd

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).

    Parameters:
        close (pd.Series): Series of closing prices.
        period (int): Lookback period for RSI calculation.

    Returns:
        pd.Series: RSI values.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
