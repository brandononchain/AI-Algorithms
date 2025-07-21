# %% [markdown]
"""
# Hyperparameter Optimization with Optuna

Tune agent parameters for best backtest performance.
"""

# %% [code]
import optuna
from utils.performance import sharpe_ratio
from research.backtest_engine import VectorBacktester

# %% [code]
def objective(trial):
    lookback = trial.suggest_int("lookback", 5, 60)
    threshold = trial.suggest_float("z_threshold", 0.5, 3.0)
    # generate signals...
    signals = ...  # user code
    df_bt = VectorBacktester(df).apply_signal(signals)
    return sharpe_ratio(df_bt["equity_curve"].pct_change().dropna())

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
