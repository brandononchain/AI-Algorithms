# %% [markdown]
"""
# Explainability with SHAP

Use SHAP to interpret ML‐based strategies.
"""

# %% [code]
import shap
from utils.ml_utils import build_features
from sklearn.ensemble import RandomForestClassifier

X, y = build_features(df)
model = RandomForestClassifier().fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
# %% [code]
shap.summary_plot(shap_values, X)
