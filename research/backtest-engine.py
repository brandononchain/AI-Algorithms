# %% [markdown]
"""
# Enhanced Vectorized Backtesting Engine

Comprehensive backtesting system with portfolio management, transaction costs, 
slippage, risk management, and detailed performance analytics.
"""

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# %% [code]
@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005   # 0.05% slippage
    max_leverage: float = 1.0
    position_sizing: str = 'fixed'  # 'fixed', 'percent_risk', 'kelly'
    risk_per_trade: float = 0.02   # 2% risk per trade
    max_positions: int = 10
    margin_requirement: float = 0.1  # 10% margin for leveraged positions

# %% [code]
class EnhancedBacktester:
    """
    Enhanced backtesting engine with comprehensive features:
    - Portfolio management
    - Transaction costs and slippage
    - Position sizing strategies
    - Risk management
    - Detailed performance metrics
    """
    
    def __init__(self, data: pd.DataFrame, config: BacktestConfig = None):
        self.data = data.copy()
        self.config = config or BacktestConfig()
        self.reset()
        
    def reset(self):
        """Reset the backtester state"""
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions = pd.Series(0.0, index=self.data.index)
        self.trades = []
        self.portfolio_history = []
        self.returns = self.data['close'].pct_change().fillna(0)
        
    def calculate_position_size(self, price: float, signal_strength: float = 1.0, 
                              volatility: float = None) -> float:
        """Calculate position size based on configuration"""
        if self.config.position_sizing == 'fixed':
            return self.config.initial_capital * 0.1  # 10% of capital
        elif self.config.position_sizing == 'percent_risk':
            if volatility is None:
                volatility = self.returns.rolling(20).std().iloc[-1]
            risk_amount = self.portfolio_value * self.config.risk_per_trade
            return risk_amount / (volatility * price)
        elif self.config.position_sizing == 'kelly':
            # Simplified Kelly criterion implementation
            win_rate = 0.55  # This should be estimated from historical performance
            avg_win_loss_ratio = 1.2  # This should be estimated from historical performance
            kelly_fraction = win_rate - (1 - win_rate) / avg_win_loss_ratio
            return self.portfolio_value * min(kelly_fraction * signal_strength, 0.25)
        return self.config.initial_capital * 0.1
    
    def apply_transaction_costs(self, trade_value: float) -> float:
        """Apply commission and slippage to trade"""
        commission_cost = abs(trade_value) * self.config.commission
        slippage_cost = abs(trade_value) * self.config.slippage
        return commission_cost + slippage_cost
    
    def backtest_strategy(self, signals: pd.Series, signal_strength: pd.Series = None) -> Dict:
        """
        Run backtest with given signals
        
        Args:
            signals: Trading signals (-1, 0, 1)
            signal_strength: Optional signal strength (0-1)
        """
        if signal_strength is None:
            signal_strength = pd.Series(1.0, index=signals.index)
            
        portfolio_values = []
        cash_values = []
        position_values = []
        current_position = 0.0
        
        for i, (timestamp, signal) in enumerate(signals.items()):
            if i == 0:
                portfolio_values.append(self.portfolio_value)
                cash_values.append(self.cash)
                position_values.append(0.0)
                continue
                
            price = self.data.loc[timestamp, 'close']
            prev_price = self.data.iloc[i-1]['close'] if i > 0 else price
            
            # Update portfolio value based on price changes
            if current_position != 0:
                price_change = (price - prev_price) / prev_price
                position_pnl = current_position * price_change * prev_price
                self.portfolio_value += position_pnl
            
            # Handle new signals
            if signal != 0 and signal != current_position:
                # Close existing position
                if current_position != 0:
                    trade_value = current_position * price
                    transaction_cost = self.apply_transaction_costs(trade_value)
                    self.cash += trade_value - transaction_cost
                    
                    # Record trade
                    self.trades.append({
                        'timestamp': timestamp,
                        'type': 'CLOSE',
                        'size': -current_position,
                        'price': price,
                        'value': trade_value,
                        'cost': transaction_cost
                    })
                    current_position = 0.0
                
                # Open new position
                if signal != 0:
                    strength = signal_strength.loc[timestamp]
                    position_size = self.calculate_position_size(price, strength)
                    position_size *= signal  # Apply signal direction
                    
                    # Check if we have enough cash/margin
                    required_cash = abs(position_size * price)
                    if self.config.max_leverage > 1:
                        required_cash *= self.config.margin_requirement
                    
                    if required_cash <= self.cash:
                        trade_value = position_size * price
                        transaction_cost = self.apply_transaction_costs(trade_value)
                        self.cash -= required_cash + transaction_cost
                        current_position = position_size
                        
                        # Record trade
                        self.trades.append({
                            'timestamp': timestamp,
                            'type': 'OPEN',
                            'size': position_size,
                            'price': price,
                            'value': trade_value,
                            'cost': transaction_cost
                        })
            
            # Update portfolio tracking
            position_value = current_position * price if current_position != 0 else 0.0
            self.portfolio_value = self.cash + position_value
            
            portfolio_values.append(self.portfolio_value)
            cash_values.append(self.cash)
            position_values.append(position_value)
            self.positions.iloc[i] = current_position
        
        # Create results DataFrame
        results = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'cash': cash_values,
            'position_value': position_values,
            'positions': self.positions,
            'returns': pd.Series(portfolio_values).pct_change().fillna(0),
            'price': self.data['close']
        }, index=self.data.index)
        
        return self._calculate_performance_metrics(results)
    
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        returns = results['returns']
        portfolio_values = results['portfolio_value']
        
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
        
        # Trade analysis
        num_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        # Additional metrics
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'results_df': results,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_values.iloc[-1],
            'trades': self.trades
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - 0.03/252  # Assuming 3% risk-free rate
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return np.inf
        downside_deviation = negative_returns.std() * np.sqrt(252)
        return excess_returns.mean() * np.sqrt(252) / downside_deviation
    
    def plot_results(self, results: Dict, figsize: Tuple[int, int] = (15, 10)):
        """Plot comprehensive backtesting results"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        results_df = results['results_df']
        
        # Portfolio value over time
        axes[0, 0].plot(results_df.index, results_df['portfolio_value'], 
                       label='Portfolio Value', linewidth=2)
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
        axes[0, 1].set_title(f'Drawdown (Max: {results["max_drawdown"]:.2%})')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[1, 0].hist(results_df['returns'] * 100, bins=50, alpha=0.7, 
                       edgecolor='black')
        axes[1, 0].axvline(results_df['returns'].mean() * 100, color='red', 
                          linestyle='--', label=f'Mean: {results_df["returns"].mean()*100:.3f}%')
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        rolling_sharpe = results_df['returns'].rolling(60).mean() / results_df['returns'].rolling(60).std() * np.sqrt(252)
        axes[1, 1].plot(results_df.index, rolling_sharpe, linewidth=2)
        axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Sharpe = 1')
        axes[1, 1].set_title(f'60-Day Rolling Sharpe Ratio (Final: {results["sharpe_ratio"]:.2f})')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance summary
        self._print_performance_summary(results)
    
    def _print_performance_summary(self, results: Dict):
        """Print formatted performance summary"""
        print("=" * 60)
        print("BACKTESTING PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Initial Capital:      ${self.config.initial_capital:,.2f}")
        print(f"Final Portfolio:      ${results['final_portfolio_value']:,.2f}")
        print(f"Total Return:         {results['total_return']:.2%}")
        print(f"Annualized Return:    {results['annualized_return']:.2%}")
        print(f"Volatility:           {results['volatility']:.2%}")
        print(f"Sharpe Ratio:         {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:        {results['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:         {results['calmar_ratio']:.2f}")
        print(f"Maximum Drawdown:     {results['max_drawdown']:.2%}")
        print(f"Number of Trades:     {results['num_trades']}")
        print(f"Win Rate:             {results['win_rate']:.2%}")
        print("=" * 60)

# %% [code]
class StrategyComparator:
    """Compare multiple strategies side by side"""
    
    def __init__(self, data: pd.DataFrame, config: BacktestConfig = None):
        self.data = data
        self.config = config or BacktestConfig()
        self.results = {}
    
    def add_strategy(self, name: str, signals: pd.Series, signal_strength: pd.Series = None):
        """Add a strategy for comparison"""
        backtester = EnhancedBacktester(self.data, self.config)
        result = backtester.backtest_strategy(signals, signal_strength)
        self.results[name] = result
    
    def compare_strategies(self) -> pd.DataFrame:
        """Create comparison table of all strategies"""
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Strategy': name,
                'Total Return': f"{result['total_return']:.2%}",
                'Ann. Return': f"{result['annualized_return']:.2%}",
                'Volatility': f"{result['volatility']:.2%}",
                'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{result['max_drawdown']:.2%}",
                'Num Trades': result['num_trades'],
                'Win Rate': f"{result['win_rate']:.2%}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_comparison(self, figsize: Tuple[int, int] = (15, 8)):
        """Plot comparison of strategy performance"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Portfolio values
        for name, result in self.results.items():
            results_df = result['results_df']
            axes[0].plot(results_df.index, results_df['portfolio_value'], 
                        label=name, linewidth=2)
        
        axes[0].axhline(y=self.config.initial_capital, color='black', 
                       linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0].set_title('Portfolio Value Comparison')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Risk-Return scatter
        returns = [result['annualized_return'] for result in self.results.values()]
        volatilities = [result['volatility'] for result in self.results.values()]
        names = list(self.results.keys())
        
        axes[1].scatter(volatilities, returns, s=100, alpha=0.7)
        for i, name in enumerate(names):
            axes[1].annotate(name, (volatilities[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points')
        
        axes[1].set_xlabel('Volatility')
        axes[1].set_ylabel('Annualized Return')
        axes[1].set_title('Risk-Return Profile')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# %% [code]
# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
    
    sample_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Create simple moving average crossover signals
    short_ma = sample_data['close'].rolling(10).mean()
    long_ma = sample_data['close'].rolling(30).mean()
    signals = pd.Series(0, index=sample_data.index)
    signals[short_ma > long_ma] = 1
    signals[short_ma < long_ma] = -1
    
    # Run backtest
    config = BacktestConfig(initial_capital=100000, commission=0.001)
    backtester = EnhancedBacktester(sample_data, config)
    results = backtester.backtest_strategy(signals)
    
    # Display results
    backtester.plot_results(results)
