import pandas as pd

def calculate_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Parameters:
        close (pd.Series): Series of closing prices.
        window (int): Rolling window for the moving average.
        num_std (float): Number of standard deviations for the bands.

    Returns:
        pd.DataFrame: DataFrame with columns ['middle_band', 'upper_band', 'lower_band'].
    """
    middle_band = close.rolling(window).mean()
    std = close.rolling(window).std()

    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)

    return pd.DataFrame({
        'middle_band': middle_band,
        'upper_band': upper_band,
        'lower_band': lower_band
    })
