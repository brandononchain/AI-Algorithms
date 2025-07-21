import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_feat = pd.DataFrame(index=df.index)
    df_feat["rsi"] = ...        # compute RSI
    df_feat["macd_diff"] = ...  # compute MACD histogram
    df_feat["atr"] = ...        # compute ATR
    # label next‐bar direction
    y = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    return df_feat.dropna(), pd.Series(y[df_feat.dropna().index])
