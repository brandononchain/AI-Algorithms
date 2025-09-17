"""
Portfolio Management System

Multi-strategy portfolio management with:
- Dynamic allocation
- Risk budgeting
- Correlation management
- Performance attribution
- Rebalancing strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management"""
    initial_capital: float = 1000000.0
    max_strategy_weight: float = 0.4  # Maximum weight per strategy
    min_strategy_weight: float = 0.05  # Minimum weight per strategy
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    risk_budget_method: str = 'equal_risk'  # 'equal_weight', 'equal_risk', 'risk_parity', 'mean_variance'
    max_correlation: float = 0.8  # Maximum correlation between strategies
    volatility_target: float = 0.15  # Target portfolio volatility
    max_drawdown_limit: float = 0.15  # Maximum allowed drawdown
    transaction_cost: float = 0.001  # Transaction cost for rebalancing


class PortfolioManager:
    """
    Advanced portfolio management system for multi-strategy allocation.
    """
    
    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        self.strategies = {}
        self.weights = {}
        self.portfolio_history = []
        self.rebalance_dates = []
        self.transaction_costs = []
        
    def add_strategy(self, name: str, returns: pd.Series, 
                    strategy_type: str = "unknown", 
                    benchmark: pd.Series = None):
        """Add a strategy to the portfolio"""
        self.strategies[name] = {
            'returns': returns,
            'type': strategy_type,
            'benchmark': benchmark,
            'sharpe_ratio': self._calculate_sharpe(returns),
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(returns.cumsum())
        }
        
    def _calculate_sharpe(self, returns: pd.Series, rf_rate: float = 0.03) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - rf_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def calculate_correlation_matrix(self, lookback_days: int = 252) -> pd.DataFrame:
        """Calculate correlation matrix of strategy returns"""
        strategy_names = list(self.strategies.keys())
        returns_df = pd.DataFrame()
        
        for name in strategy_names:
            returns_df[name] = self.strategies[name]['returns']
        
        # Use rolling correlation if specified
        if lookback_days:
            correlation_matrix = returns_df.tail(lookback_days).corr()
        else:
            correlation_matrix = returns_df.corr()
            
        return correlation_matrix
    
    def optimize_weights_equal_risk(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Equal risk contribution optimization"""
        n_assets = len(returns_df.columns)
        cov_matrix = returns_df.cov() * 252  # Annualized covariance
        
        def risk_budget_objective(weights, cov_matrix):
            """Objective function for equal risk contribution"""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((self.config.min_strategy_weight, self.config.max_strategy_weight) 
                      for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(risk_budget_objective, x0, 
                         args=(cov_matrix,), method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def optimize_weights_mean_variance(self, returns_df: pd.DataFrame, 
                                     target_return: float = None) -> np.ndarray:
        """Mean-variance optimization"""
        n_assets = len(returns_df.columns)
        mean_returns = returns_df.mean() * 252  # Annualized returns
        cov_matrix = returns_df.cov() * 252  # Annualized covariance
        
        if target_return is None:
            target_return = mean_returns.mean()
        
        def portfolio_variance(weights, cov_matrix):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return}  # Target return
        ]
        
        # Bounds
        bounds = tuple((self.config.min_strategy_weight, self.config.max_strategy_weight) 
                      for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(portfolio_variance, x0, 
                         args=(cov_matrix,), method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def optimize_weights_max_diversification(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Maximum diversification optimization"""
        n_assets = len(returns_df.columns)
        volatilities = returns_df.std() * np.sqrt(252)  # Annualized volatilities
        cov_matrix = returns_df.cov() * 252
        
        def diversification_ratio(weights, volatilities, cov_matrix):
            """Diversification ratio to maximize"""
            weighted_vol = np.dot(weights, volatilities)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -weighted_vol / portfolio_vol  # Negative for maximization
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((self.config.min_strategy_weight, self.config.max_strategy_weight) 
                      for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(diversification_ratio, x0, 
                         args=(volatilities, cov_matrix), method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def calculate_optimal_weights(self, rebalance_date: pd.Timestamp, 
                                lookback_days: int = 252) -> Dict[str, float]:
        """Calculate optimal portfolio weights"""
        strategy_names = list(self.strategies.keys())
        
        if not strategy_names:
            return {}
        
        # Create returns DataFrame for optimization
        returns_df = pd.DataFrame()
        for name in strategy_names:
            strategy_returns = self.strategies[name]['returns']
            # Get returns up to rebalance date
            available_returns = strategy_returns[strategy_returns.index <= rebalance_date]
            if len(available_returns) >= lookback_days:
                returns_df[name] = available_returns.tail(lookback_days)
        
        if returns_df.empty or len(returns_df.columns) == 0:
            # Equal weights as fallback
            equal_weight = 1.0 / len(strategy_names)
            return {name: equal_weight for name in strategy_names}
        
        # Remove strategies with insufficient data
        valid_strategies = returns_df.columns.tolist()
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 60:  # Minimum data requirement
            equal_weight = 1.0 / len(valid_strategies)
            return {name: equal_weight for name in valid_strategies}
        
        # Check correlations and remove highly correlated strategies
        corr_matrix = returns_df.corr()
        to_remove = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.config.max_correlation:
                    # Remove strategy with lower Sharpe ratio
                    strategy1 = corr_matrix.columns[i]
                    strategy2 = corr_matrix.columns[j]
                    
                    sharpe1 = self.strategies[strategy1]['sharpe_ratio']
                    sharpe2 = self.strategies[strategy2]['sharpe_ratio']
                    
                    if sharpe1 < sharpe2:
                        to_remove.add(strategy1)
                    else:
                        to_remove.add(strategy2)
        
        # Remove highly correlated strategies
        final_strategies = [s for s in valid_strategies if s not in to_remove]
        if not final_strategies:
            final_strategies = valid_strategies[:1]  # Keep at least one strategy
        
        returns_df = returns_df[final_strategies]
        
        # Optimize weights based on method
        if self.config.risk_budget_method == 'equal_weight':
            weights = np.array([1.0 / len(final_strategies)] * len(final_strategies))
        elif self.config.risk_budget_method == 'equal_risk':
            weights = self.optimize_weights_equal_risk(returns_df)
        elif self.config.risk_budget_method == 'mean_variance':
            weights = self.optimize_weights_mean_variance(returns_df)
        elif self.config.risk_budget_method == 'max_diversification':
            weights = self.optimize_weights_max_diversification(returns_df)
        else:
            weights = np.array([1.0 / len(final_strategies)] * len(final_strategies))
        
        # Create weights dictionary
        weight_dict = {}
        for i, strategy in enumerate(final_strategies):
            weight_dict[strategy] = weights[i]
        
        # Add zero weights for removed strategies
        for strategy in strategy_names:
            if strategy not in weight_dict:
                weight_dict[strategy] = 0.0
        
        return weight_dict
    
    def get_rebalance_dates(self, start_date: pd.Timestamp, 
                          end_date: pd.Timestamp) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency"""
        dates = []
        current_date = start_date
        
        if self.config.rebalance_frequency == 'daily':
            dates = pd.date_range(start_date, end_date, freq='D').tolist()
        elif self.config.rebalance_frequency == 'weekly':
            dates = pd.date_range(start_date, end_date, freq='W').tolist()
        elif self.config.rebalance_frequency == 'monthly':
            dates = pd.date_range(start_date, end_date, freq='M').tolist()
        elif self.config.rebalance_frequency == 'quarterly':
            dates = pd.date_range(start_date, end_date, freq='Q').tolist()
        
        return [pd.Timestamp(date) for date in dates]
    
    def calculate_transaction_costs(self, old_weights: Dict[str, float], 
                                  new_weights: Dict[str, float], 
                                  portfolio_value: float) -> float:
        """Calculate transaction costs for rebalancing"""
        total_turnover = 0.0
        
        for strategy in set(list(old_weights.keys()) + list(new_weights.keys())):
            old_weight = old_weights.get(strategy, 0.0)
            new_weight = new_weights.get(strategy, 0.0)
            total_turnover += abs(new_weight - old_weight)
        
        return total_turnover * portfolio_value * self.config.transaction_cost
    
    def backtest_portfolio(self, start_date: pd.Timestamp = None, 
                         end_date: pd.Timestamp = None) -> Dict:
        """Backtest the multi-strategy portfolio"""
        if not self.strategies:
            raise ValueError("No strategies added to portfolio")
        
        # Get common date range
        all_dates = set()
        for strategy in self.strategies.values():
            all_dates.update(strategy['returns'].index)
        
        all_dates = sorted(list(all_dates))
        
        if start_date is None:
            start_date = pd.Timestamp(all_dates[252])  # Skip first year for warmup
        if end_date is None:
            end_date = pd.Timestamp(all_dates[-1])
        
        # Filter dates
        backtest_dates = [date for date in all_dates if start_date <= date <= end_date]
        
        # Get rebalancing dates
        rebalance_dates = self.get_rebalance_dates(start_date, end_date)
        rebalance_dates = [date for date in rebalance_dates if date in backtest_dates]
        
        # Initialize portfolio
        portfolio_value = self.config.initial_capital
        current_weights = {}
        portfolio_returns = []
        portfolio_values = [portfolio_value]
        weights_history = []
        transaction_costs_history = []
        
        for i, date in enumerate(backtest_dates):
            # Check if rebalancing is needed
            if date in rebalance_dates or not current_weights:
                old_weights = current_weights.copy()
                new_weights = self.calculate_optimal_weights(date)
                
                # Calculate transaction costs
                if old_weights:
                    transaction_cost = self.calculate_transaction_costs(
                        old_weights, new_weights, portfolio_value
                    )
                    portfolio_value -= transaction_cost
                    transaction_costs_history.append(transaction_cost)
                else:
                    transaction_costs_history.append(0.0)
                
                current_weights = new_weights
                weights_history.append((date, current_weights.copy()))
            
            # Calculate portfolio return for this period
            portfolio_return = 0.0
            for strategy_name, weight in current_weights.items():
                if weight > 0:
                    strategy_returns = self.strategies[strategy_name]['returns']
                    if date in strategy_returns.index:
                        strategy_return = strategy_returns[date]
                        portfolio_return += weight * strategy_return
            
            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)
            portfolio_returns.append(portfolio_return)
            portfolio_values.append(portfolio_value)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'portfolio_value': portfolio_values[1:],  # Skip initial value
            'portfolio_returns': portfolio_returns
        }, index=backtest_dates)
        
        # Calculate performance metrics
        portfolio_returns_series = pd.Series(portfolio_returns, index=backtest_dates)
        
        performance_metrics = self._calculate_portfolio_metrics(
            results_df, portfolio_returns_series
        )
        
        return {
            'results_df': results_df,
            'performance_metrics': performance_metrics,
            'weights_history': weights_history,
            'transaction_costs': sum(transaction_costs_history),
            'rebalance_dates': rebalance_dates
        }
    
    def _calculate_portfolio_metrics(self, results_df: pd.DataFrame, 
                                   returns: pd.Series) -> Dict:
        """Calculate comprehensive portfolio performance metrics"""
        portfolio_values = results_df['portfolio_value']
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / self.config.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.03) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'final_portfolio_value': portfolio_values.iloc[-1]
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - 0.03/252
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return np.inf
        downside_deviation = negative_returns.std() * np.sqrt(252)
        return excess_returns.mean() * np.sqrt(252) / downside_deviation
    
    def plot_portfolio_performance(self, backtest_results: Dict, 
                                 figsize: Tuple[int, int] = (15, 12)):
        """Plot comprehensive portfolio performance analysis"""
        results_df = backtest_results['results_df']
        weights_history = backtest_results['weights_history']
        metrics = backtest_results['performance_metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Portfolio value over time
        axes[0, 0].plot(results_df.index, results_df['portfolio_value'], 
                       linewidth=2, label='Portfolio Value')
        axes[0, 0].axhline(y=self.config.initial_capital, color='r', 
                          linestyle='--', alpha=0.7, label='Initial Capital')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = results_df['portfolio_value'].expanding().max()
        drawdown = (results_df['portfolio_value'] - rolling_max) / rolling_max * 100
        axes[0, 1].fill_between(results_df.index, drawdown, 0, 
                               color='red', alpha=0.3)
        axes[0, 1].plot(results_df.index, drawdown, color='red', linewidth=1)
        axes[0, 1].set_title(f'Drawdown (Max: {metrics["max_drawdown"]:.2%})')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[1, 0].hist(results_df['portfolio_returns'] * 100, bins=50, 
                       alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(results_df['portfolio_returns'].mean() * 100, 
                          color='red', linestyle='--', 
                          label=f'Mean: {results_df["portfolio_returns"].mean()*100:.3f}%')
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Strategy weights over time
        if weights_history:
            strategy_names = list(self.strategies.keys())
            weight_dates = [item[0] for item in weights_history]
            
            for strategy in strategy_names:
                weights = [item[1].get(strategy, 0) for item in weights_history]
                axes[1, 1].plot(weight_dates, weights, marker='o', 
                               label=strategy, alpha=0.7)
            
            axes[1, 1].set_title('Strategy Weights Over Time')
            axes[1, 1].set_ylabel('Weight')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance summary
        self._print_portfolio_summary(metrics, backtest_results)
    
    def _print_portfolio_summary(self, metrics: Dict, backtest_results: Dict):
        """Print formatted portfolio performance summary"""
        print("=" * 70)
        print("MULTI-STRATEGY PORTFOLIO PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"Initial Capital:        ${self.config.initial_capital:,.2f}")
        print(f"Final Portfolio Value:  ${metrics['final_portfolio_value']:,.2f}")
        print(f"Total Return:           {metrics['total_return']:.2%}")
        print(f"Annualized Return:      {metrics['annualized_return']:.2%}")
        print(f"Volatility:             {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:          {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:           {metrics['calmar_ratio']:.2f}")
        print(f"Maximum Drawdown:       {metrics['max_drawdown']:.2%}")
        print(f"VaR (95%):              {metrics['var_95']:.2%}")
        print(f"CVaR (95%):             {metrics['cvar_95']:.2%}")
        print(f"Transaction Costs:      ${backtest_results['transaction_costs']:,.2f}")
        print(f"Number of Rebalances:   {len(backtest_results['rebalance_dates'])}")
        print("=" * 70)
        
        # Strategy contribution analysis
        print("\nSTRATEGY ANALYSIS:")
        print("-" * 40)
        for name, strategy in self.strategies.items():
            print(f"{name}:")
            print(f"  Sharpe Ratio:     {strategy['sharpe_ratio']:.2f}")
            print(f"  Volatility:       {strategy['volatility']:.2%}")
            print(f"  Max Drawdown:     {strategy['max_drawdown']:.2%}")
            print()


# Example usage
if __name__ == "__main__":
    # Generate sample strategy returns
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create different strategy return patterns
    trend_following = np.random.randn(len(dates)) * 0.01 + 0.0003  # Slight positive drift
    mean_reversion = np.random.randn(len(dates)) * 0.008 + 0.0002
    momentum = np.random.randn(len(dates)) * 0.012 + 0.0004
    volatility = np.random.randn(len(dates)) * 0.015 + 0.0001
    
    # Add some correlation structure
    common_factor = np.random.randn(len(dates)) * 0.005
    trend_following += 0.3 * common_factor
    momentum += 0.4 * common_factor
    
    # Convert to pandas Series
    strategy_returns = {
        'Trend Following': pd.Series(trend_following, index=dates),
        'Mean Reversion': pd.Series(mean_reversion, index=dates),
        'Momentum': pd.Series(momentum, index=dates),
        'Volatility': pd.Series(volatility, index=dates)
    }
    
    # Create portfolio manager
    config = PortfolioConfig(
        initial_capital=1000000,
        rebalance_frequency='monthly',
        risk_budget_method='equal_risk',
        max_strategy_weight=0.6,
        min_strategy_weight=0.1
    )
    
    portfolio_manager = PortfolioManager(config)
    
    # Add strategies
    for name, returns in strategy_returns.items():
        portfolio_manager.add_strategy(name, returns, name.lower().replace(' ', '_'))
    
    # Backtest portfolio
    backtest_results = portfolio_manager.backtest_portfolio()
    
    # Display results
    portfolio_manager.plot_portfolio_performance(backtest_results)