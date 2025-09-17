"""
Strategy Optimization and Walk-Forward Analysis

Advanced parameter optimization system:
- Grid search and random search
- Bayesian optimization
- Genetic algorithms
- Walk-forward analysis
- Monte Carlo simulation
- Overfitting detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from itertools import product
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import optuna
from datetime import datetime, timedelta
import pickle
import json

warnings.filterwarnings('ignore')


@dataclass
class OptimizationConfig:
    """Configuration for strategy optimization"""
    # Optimization method
    method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian', 'genetic', 'optuna'
    
    # Search parameters
    max_iterations: int = 100
    n_random_starts: int = 10
    cv_folds: int = 5
    
    # Objective function
    objective_metric: str = 'sharpe_ratio'  # 'sharpe_ratio', 'calmar_ratio', 'sortino_ratio', 'total_return'
    maximize: bool = True
    
    # Walk-forward analysis
    training_window: int = 252  # Trading days for training
    testing_window: int = 63   # Trading days for testing
    step_size: int = 21        # Days to step forward
    min_trades: int = 10       # Minimum trades required
    
    # Overfitting detection
    max_sharpe_threshold: float = 3.0  # Flag potentially overfit strategies
    consistency_threshold: float = 0.7  # Minimum consistency across periods
    
    # Parallel processing
    n_jobs: int = -1  # Number of parallel jobs
    
    # Storage
    save_results: bool = True
    results_directory: str = './optimization_results'


class ParameterSpace:
    """Define parameter search space"""
    
    def __init__(self):
        self.parameters = {}
        self.constraints = []
    
    def add_parameter(self, name: str, param_type: str, **kwargs):
        """Add a parameter to the search space"""
        self.parameters[name] = {
            'type': param_type,
            **kwargs
        }
    
    def add_constraint(self, constraint_func: Callable):
        """Add a constraint function"""
        self.constraints.append(constraint_func)
    
    def generate_grid(self) -> List[Dict]:
        """Generate parameter grid for grid search"""
        param_lists = {}
        
        for name, param_info in self.parameters.items():
            if param_info['type'] == 'discrete':
                param_lists[name] = param_info['values']
            elif param_info['type'] == 'continuous':
                start, end, step = param_info['min'], param_info['max'], param_info.get('step', 0.1)
                param_lists[name] = np.arange(start, end + step, step).tolist()
            elif param_info['type'] == 'integer':
                start, end = param_info['min'], param_info['max']
                param_lists[name] = list(range(start, end + 1))
        
        # Generate all combinations
        grid = list(ParameterGrid(param_lists))
        
        # Apply constraints
        if self.constraints:
            filtered_grid = []
            for params in grid:
                if all(constraint(params) for constraint in self.constraints):
                    filtered_grid.append(params)
            return filtered_grid
        
        return grid
    
    def sample_random(self, n_samples: int) -> List[Dict]:
        """Generate random parameter samples"""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            for name, param_info in self.parameters.items():
                if param_info['type'] == 'discrete':
                    sample[name] = np.random.choice(param_info['values'])
                elif param_info['type'] == 'continuous':
                    sample[name] = np.random.uniform(param_info['min'], param_info['max'])
                elif param_info['type'] == 'integer':
                    sample[name] = np.random.randint(param_info['min'], param_info['max'] + 1)
            
            # Check constraints
            if not self.constraints or all(constraint(sample) for constraint in self.constraints):
                samples.append(sample)
        
        return samples


class ObjectiveFunction:
    """Objective function for optimization"""
    
    def __init__(self, strategy_func: Callable, backtest_func: Callable, 
                 data: pd.DataFrame, config: OptimizationConfig):
        self.strategy_func = strategy_func
        self.backtest_func = backtest_func
        self.data = data
        self.config = config
        self.evaluation_cache = {}
    
    def evaluate(self, parameters: Dict) -> float:
        """Evaluate strategy with given parameters"""
        # Create cache key
        cache_key = tuple(sorted(parameters.items()))
        
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        try:
            # Generate signals with parameters
            strategy = self.strategy_func(parameters)
            signals = strategy.generate_detailed_signals(self.data)
            
            if signals is None or 'signal' not in signals.columns:
                return -np.inf if self.config.maximize else np.inf
            
            # Backtest strategy
            results = self.backtest_func(self.data, signals['signal'])
            
            if not results or 'performance_metrics' not in results:
                return -np.inf if self.config.maximize else np.inf
            
            # Extract objective metric
            metrics = results['performance_metrics']
            objective_value = metrics.get(self.config.objective_metric, 0)
            
            # Apply constraints
            num_trades = metrics.get('num_trades', 0)
            if num_trades < self.config.min_trades:
                objective_value = -np.inf if self.config.maximize else np.inf
            
            # Check for overfitting (unrealistic Sharpe ratios)
            if (self.config.objective_metric == 'sharpe_ratio' and 
                objective_value > self.config.max_sharpe_threshold):
                objective_value = self.config.max_sharpe_threshold
            
            # Cache result
            self.evaluation_cache[cache_key] = objective_value
            
            return objective_value
            
        except Exception as e:
            print(f"Error evaluating parameters {parameters}: {e}")
            return -np.inf if self.config.maximize else np.inf


class GridSearchOptimizer:
    """Grid search optimization"""
    
    def __init__(self, objective_func: ObjectiveFunction, config: OptimizationConfig):
        self.objective_func = objective_func
        self.config = config
    
    def optimize(self, parameter_space: ParameterSpace) -> Dict:
        """Run grid search optimization"""
        grid = parameter_space.generate_grid()
        print(f"Grid search: evaluating {len(grid)} parameter combinations")
        
        results = []
        
        if self.config.n_jobs == 1:
            # Sequential execution
            for i, params in enumerate(grid):
                score = self.objective_func.evaluate(params)
                results.append({
                    'parameters': params,
                    'score': score,
                    'iteration': i
                })
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(grid)} evaluations")
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                future_to_params = {
                    executor.submit(self.objective_func.evaluate, params): (i, params)
                    for i, params in enumerate(grid)
                }
                
                for future in as_completed(future_to_params):
                    i, params = future_to_params[future]
                    try:
                        score = future.result()
                        results.append({
                            'parameters': params,
                            'score': score,
                            'iteration': i
                        })
                    except Exception as exc:
                        print(f"Parameter evaluation generated an exception: {exc}")
                        results.append({
                            'parameters': params,
                            'score': -np.inf if self.config.maximize else np.inf,
                            'iteration': i
                        })
                    
                    if len(results) % 10 == 0:
                        print(f"Completed {len(results)}/{len(grid)} evaluations")
        
        # Sort results
        results.sort(key=lambda x: x['score'], reverse=self.config.maximize)
        
        return {
            'best_parameters': results[0]['parameters'],
            'best_score': results[0]['score'],
            'all_results': results,
            'method': 'grid_search'
        }


class BayesianOptimizer:
    """Bayesian optimization using Gaussian Process"""
    
    def __init__(self, objective_func: ObjectiveFunction, config: OptimizationConfig):
        self.objective_func = objective_func
        self.config = config
    
    def optimize(self, parameter_space: ParameterSpace) -> Dict:
        """Run Bayesian optimization"""
        # Convert parameter space to bounds
        bounds = []
        param_names = []
        
        for name, param_info in parameter_space.parameters.items():
            if param_info['type'] in ['continuous', 'integer']:
                bounds.append((param_info['min'], param_info['max']))
                param_names.append(name)
        
        if not bounds:
            raise ValueError("Bayesian optimization requires continuous or integer parameters")
        
        # Initialize with random samples
        X_init = []
        y_init = []
        
        for _ in range(self.config.n_random_starts):
            params = {}
            for i, name in enumerate(param_names):
                param_info = parameter_space.parameters[name]
                if param_info['type'] == 'continuous':
                    params[name] = np.random.uniform(bounds[i][0], bounds[i][1])
                else:  # integer
                    params[name] = np.random.randint(bounds[i][0], bounds[i][1] + 1)
            
            # Add discrete parameters if any
            for name, param_info in parameter_space.parameters.items():
                if param_info['type'] == 'discrete':
                    params[name] = np.random.choice(param_info['values'])
            
            score = self.objective_func.evaluate(params)
            X_init.append([params[name] for name in param_names])
            y_init.append(score)
        
        X_init = np.array(X_init)
        y_init = np.array(y_init)
        
        # Gaussian Process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        
        best_params = None
        best_score = -np.inf if self.config.maximize else np.inf
        all_results = []
        
        for iteration in range(self.config.max_iterations - self.config.n_random_starts):
            # Fit GP
            gp.fit(X_init, y_init)
            
            # Acquisition function (Expected Improvement)
            def acquisition(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                
                if self.config.maximize:
                    improvement = mu - np.max(y_init)
                    Z = improvement / sigma
                    ei = improvement * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
                    return -ei  # Minimize negative EI
                else:
                    improvement = np.min(y_init) - mu
                    Z = improvement / sigma
                    ei = improvement * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
                    return -ei
            
            # Optimize acquisition function
            best_x = None
            best_acq = np.inf
            
            for _ in range(100):  # Multiple random starts
                x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
                res = minimize(acquisition, x0, bounds=bounds, method='L-BFGS-B')
                
                if res.fun < best_acq:
                    best_acq = res.fun
                    best_x = res.x
            
            # Convert back to parameter dict
            next_params = {}
            for i, name in enumerate(param_names):
                param_info = parameter_space.parameters[name]
                if param_info['type'] == 'integer':
                    next_params[name] = int(round(best_x[i]))
                else:
                    next_params[name] = best_x[i]
            
            # Add discrete parameters
            for name, param_info in parameter_space.parameters.items():
                if param_info['type'] == 'discrete':
                    next_params[name] = np.random.choice(param_info['values'])
            
            # Evaluate
            next_score = self.objective_func.evaluate(next_params)
            
            # Update data
            X_init = np.vstack([X_init, [next_params[name] for name in param_names]])
            y_init = np.append(y_init, next_score)
            
            # Update best
            if (self.config.maximize and next_score > best_score) or \
               (not self.config.maximize and next_score < best_score):
                best_score = next_score
                best_params = next_params.copy()
            
            all_results.append({
                'parameters': next_params,
                'score': next_score,
                'iteration': iteration + self.config.n_random_starts
            })
            
            if (iteration + 1) % 10 == 0:
                print(f"Bayesian optimization: {iteration + 1}/{self.config.max_iterations - self.config.n_random_starts} iterations")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'method': 'bayesian'
        }


class OptunaOptimizer:
    """Optuna-based optimization"""
    
    def __init__(self, objective_func: ObjectiveFunction, config: OptimizationConfig):
        self.objective_func = objective_func
        self.config = config
    
    def optimize(self, parameter_space: ParameterSpace) -> Dict:
        """Run Optuna optimization"""
        def objective(trial):
            params = {}
            
            for name, param_info in parameter_space.parameters.items():
                if param_info['type'] == 'continuous':
                    params[name] = trial.suggest_float(name, param_info['min'], param_info['max'])
                elif param_info['type'] == 'integer':
                    params[name] = trial.suggest_int(name, param_info['min'], param_info['max'])
                elif param_info['type'] == 'discrete':
                    params[name] = trial.suggest_categorical(name, param_info['values'])
            
            score = self.objective_func.evaluate(params)
            
            # Optuna maximizes by default, so negate if we want to minimize
            return score if self.config.maximize else -score
        
        # Create study
        direction = 'maximize' if self.config.maximize else 'minimize'
        study = optuna.create_study(direction=direction)
        
        # Optimize
        study.optimize(objective, n_trials=self.config.max_iterations)
        
        # Extract results
        all_results = []
        for trial in study.trials:
            score = trial.value
            if not self.config.maximize:
                score = -score  # Convert back to original scale
            
            all_results.append({
                'parameters': trial.params,
                'score': score,
                'iteration': trial.number
            })
        
        best_score = study.best_value
        if not self.config.maximize:
            best_score = -best_score
        
        return {
            'best_parameters': study.best_params,
            'best_score': best_score,
            'all_results': all_results,
            'method': 'optuna'
        }


class WalkForwardAnalyzer:
    """Walk-forward analysis for strategy validation"""
    
    def __init__(self, strategy_func: Callable, backtest_func: Callable, 
                 config: OptimizationConfig):
        self.strategy_func = strategy_func
        self.backtest_func = backtest_func
        self.config = config
    
    def run_analysis(self, data: pd.DataFrame, parameter_space: ParameterSpace) -> Dict:
        """Run walk-forward analysis"""
        results = []
        optimization_results = []
        
        # Create time windows
        data_length = len(data)
        start_idx = self.config.training_window
        
        while start_idx + self.config.testing_window < data_length:
            # Define training and testing periods
            train_start = start_idx - self.config.training_window
            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + self.config.testing_window, data_length)
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            print(f"Walk-forward period: {train_data.index[0]} to {test_data.index[-1]}")
            
            # Optimize on training data
            objective_func = ObjectiveFunction(
                self.strategy_func, self.backtest_func, train_data, self.config
            )
            
            # Use grid search for walk-forward (faster)
            optimizer = GridSearchOptimizer(objective_func, self.config)
            opt_result = optimizer.optimize(parameter_space)
            
            optimization_results.append({
                'period': (train_data.index[0], train_data.index[-1]),
                'best_parameters': opt_result['best_parameters'],
                'best_score': opt_result['best_score']
            })
            
            # Test on out-of-sample data
            best_params = opt_result['best_parameters']
            strategy = self.strategy_func(best_params)
            test_signals = strategy.generate_detailed_signals(test_data)
            
            if test_signals is not None and 'signal' in test_signals.columns:
                test_results = self.backtest_func(test_data, test_signals['signal'])
                
                if test_results and 'performance_metrics' in test_results:
                    metrics = test_results['performance_metrics']
                    
                    results.append({
                        'period': (test_data.index[0], test_data.index[-1]),
                        'parameters': best_params,
                        'metrics': metrics,
                        'in_sample_score': opt_result['best_score'],
                        'out_of_sample_score': metrics.get(self.config.objective_metric, 0)
                    })
            
            # Move to next period
            start_idx += self.config.step_size
        
        return self._analyze_walk_forward_results(results, optimization_results)
    
    def _analyze_walk_forward_results(self, results: List[Dict], 
                                    optimization_results: List[Dict]) -> Dict:
        """Analyze walk-forward results"""
        if not results:
            return {'error': 'No valid walk-forward results'}
        
        # Extract metrics
        in_sample_scores = [r['in_sample_score'] for r in results]
        out_of_sample_scores = [r['out_of_sample_score'] for r in results]
        
        # Calculate statistics
        is_mean = np.mean(in_sample_scores)
        oos_mean = np.mean(out_of_sample_scores)
        is_std = np.std(in_sample_scores)
        oos_std = np.std(out_of_sample_scores)
        
        # Overfitting metrics
        degradation = (is_mean - oos_mean) / abs(is_mean) if is_mean != 0 else 0
        consistency = np.corrcoef(in_sample_scores, out_of_sample_scores)[0, 1]
        
        # Stability metrics
        parameter_stability = self._calculate_parameter_stability(optimization_results)
        
        # Overall assessment
        is_robust = (
            degradation < 0.3 and  # Less than 30% degradation
            consistency > self.config.consistency_threshold and  # Good consistency
            oos_std / abs(oos_mean) < 2.0 if oos_mean != 0 else False  # Reasonable stability
        )
        
        return {
            'periods': len(results),
            'in_sample_mean': is_mean,
            'out_of_sample_mean': oos_mean,
            'in_sample_std': is_std,
            'out_of_sample_std': oos_std,
            'degradation': degradation,
            'consistency': consistency,
            'parameter_stability': parameter_stability,
            'is_robust': is_robust,
            'detailed_results': results,
            'optimization_history': optimization_results
        }
    
    def _calculate_parameter_stability(self, optimization_results: List[Dict]) -> float:
        """Calculate parameter stability across periods"""
        if len(optimization_results) < 2:
            return 1.0
        
        # Get all parameter names
        all_params = set()
        for result in optimization_results:
            all_params.update(result['best_parameters'].keys())
        
        # Calculate coefficient of variation for each parameter
        param_stability = {}
        
        for param in all_params:
            values = []
            for result in optimization_results:
                if param in result['best_parameters']:
                    values.append(result['best_parameters'][param])
            
            if len(values) > 1 and np.std(values) > 0:
                cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else np.inf
                param_stability[param] = 1 / (1 + cv)  # Higher is more stable
            else:
                param_stability[param] = 1.0
        
        return np.mean(list(param_stability.values()))


class StrategyOptimizer:
    """Main strategy optimization class"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
    
    def optimize_strategy(self, strategy_func: Callable, backtest_func: Callable,
                         data: pd.DataFrame, parameter_space: ParameterSpace) -> Dict:
        """Optimize strategy parameters"""
        objective_func = ObjectiveFunction(strategy_func, backtest_func, data, self.config)
        
        if self.config.method == 'grid_search':
            optimizer = GridSearchOptimizer(objective_func, self.config)
        elif self.config.method == 'bayesian':
            optimizer = BayesianOptimizer(objective_func, self.config)
        elif self.config.method == 'optuna':
            optimizer = OptunaOptimizer(objective_func, self.config)
        else:
            raise ValueError(f"Unsupported optimization method: {self.config.method}")
        
        results = optimizer.optimize(parameter_space)
        
        # Add walk-forward analysis if requested
        if hasattr(self.config, 'run_walk_forward') and self.config.run_walk_forward:
            wf_analyzer = WalkForwardAnalyzer(strategy_func, backtest_func, self.config)
            wf_results = wf_analyzer.run_analysis(data, parameter_space)
            results['walk_forward'] = wf_results
        
        return results
    
    def plot_optimization_results(self, results: Dict, figsize: Tuple[int, int] = (15, 10)):
        """Plot optimization results"""
        if 'all_results' not in results:
            print("No detailed results to plot")
            return
        
        all_results = results['all_results']
        scores = [r['score'] for r in all_results]
        iterations = [r['iteration'] for r in all_results]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Optimization progress
        axes[0, 0].plot(iterations, scores, 'b-', alpha=0.7)
        axes[0, 0].axhline(y=results['best_score'], color='r', linestyle='--', 
                          label=f"Best: {results['best_score']:.3f}")
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel(f'{self.config.objective_metric.title()}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Score distribution
        axes[0, 1].hist(scores, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(results['best_score'], color='r', linestyle='--', 
                          label=f"Best: {results['best_score']:.3f}")
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].set_xlabel(f'{self.config.objective_metric.title()}')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Parameter correlation (if applicable)
        if len(all_results) > 10:
            # Get parameter names
            param_names = list(results['best_parameters'].keys())
            if len(param_names) >= 2:
                param1, param2 = param_names[0], param_names[1]
                
                param1_values = [r['parameters'][param1] for r in all_results]
                param2_values = [r['parameters'][param2] for r in all_results]
                
                scatter = axes[1, 0].scatter(param1_values, param2_values, 
                                           c=scores, cmap='viridis', alpha=0.7)
                axes[1, 0].set_xlabel(param1)
                axes[1, 0].set_ylabel(param2)
                axes[1, 0].set_title(f'Parameter Space ({param1} vs {param2})')
                plt.colorbar(scatter, ax=axes[1, 0], label=self.config.objective_metric.title())
        
        # Walk-forward results (if available)
        if 'walk_forward' in results and 'detailed_results' in results['walk_forward']:
            wf_results = results['walk_forward']['detailed_results']
            is_scores = [r['in_sample_score'] for r in wf_results]
            oos_scores = [r['out_of_sample_score'] for r in wf_results]
            
            axes[1, 1].scatter(is_scores, oos_scores, alpha=0.7)
            axes[1, 1].plot([min(is_scores), max(is_scores)], 
                           [min(is_scores), max(is_scores)], 'r--', alpha=0.5)
            axes[1, 1].set_xlabel('In-Sample Score')
            axes[1, 1].set_ylabel('Out-of-Sample Score')
            axes[1, 1].set_title('Walk-Forward Analysis')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    from agents.momentum_agent import MomentumAgent
    from research.backtest_engine import EnhancedBacktester, BacktestConfig
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
    
    sample_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Define strategy function
    def create_momentum_strategy(params):
        return MomentumAgent(params)
    
    # Define backtest function
    def run_backtest(data, signals):
        config = BacktestConfig(initial_capital=100000)
        backtester = EnhancedBacktester(data, config)
        return backtester.backtest_strategy(signals)
    
    # Define parameter space
    param_space = ParameterSpace()
    param_space.add_parameter('fast_period', 'integer', min=5, max=20)
    param_space.add_parameter('slow_period', 'integer', min=20, max=50)
    param_space.add_parameter('momentum_threshold', 'continuous', min=0.01, max=0.05)
    
    # Add constraint: fast_period < slow_period
    param_space.add_constraint(lambda p: p['fast_period'] < p['slow_period'])
    
    # Create optimizer
    config = OptimizationConfig(
        method='grid_search',
        objective_metric='sharpe_ratio',
        max_iterations=50
    )
    
    optimizer = StrategyOptimizer(config)
    
    # Run optimization
    print("Running strategy optimization...")
    results = optimizer.optimize_strategy(
        create_momentum_strategy, run_backtest, sample_data, param_space
    )
    
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best score: {results['best_score']:.3f}")
    
    # Plot results
    optimizer.plot_optimization_results(results)