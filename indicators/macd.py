import pandas as pd

def calculate_macd(close: pd.Series, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Parameters:
        close (pd.Series): Series of closing prices.
        fast_window (int): Short-term EMA window.
        slow_window (int): Long-term EMA window.
        signal_window (int): Signal line EMA window.

    Returns:
        pd.DataFrame: DataFrame with columns ['macd', 'signal', 'histogram'].
    """
    ema_fast = close.ewm(span=fast_window, adjust=False).mean()
    ema_slow = close.ewm(span=slow_window, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })
