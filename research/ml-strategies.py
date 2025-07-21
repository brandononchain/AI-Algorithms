# %% [markdown]
"""
# ML‐Based Strategy: Random Forest

Train and backtest an RF classifier on engineered features.
"""

# %% [code]
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from utils.ml_utils import build_features

# %% [code]
X, y = build_features(df)
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Test Accuracy:", model.score(X_test, y_test))
