"""
Complete Trading System Example

This example demonstrates how to use all components of the trading system together:
1. Data loading and preprocessing
2. Strategy creation and optimization
3. Backtesting with advanced features
4. Portfolio management
5. Risk analysis
6. Comprehensive visualization

This serves as a complete end-to-end example of the trading system capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

# Import our custom modules
from utils.data_loader import DataLoader, DataConfig
from agents.momentum_agent import MomentumAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.volatility_agent import VolatilityBreakoutAgent
from research.backtest_engine import EnhancedBacktester, BacktestConfig
from research.portfolio_manager import PortfolioManager, PortfolioConfig
from research.strategy_optimizer import StrategyOptimizer, OptimizationConfig, ParameterSpace
from utils.risk_analytics import RiskAnalyzer, RiskConfig
from utils.visualization import TradingVisualizer

warnings.filterwarnings('ignore')


def main():
    """Main function demonstrating the complete trading system"""
    
    print("=" * 80)
    print("COMPLETE TRADING SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # ============================================================================
    # STEP 1: DATA LOADING AND PREPROCESSING
    # ============================================================================
    print("\n1. LOADING AND PREPROCESSING DATA")
    print("-" * 50)
    
    # Configure data loader
    data_config = DataConfig(
        start_date='2020-01-01',
        end_date='2023-12-31',
        add_technical_indicators=True,
        add_market_features=True,
        cache_data=True
    )
    
    # For this example, we'll create synthetic data since we may not have API keys
    print("Creating synthetic market data for demonstration...")
    
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Create realistic market data with trends and volatility clustering
    np.random.seed(42)
    
    # Base return process with some autocorrelation
    base_returns = np.random.randn(n_days) * 0.015
    for i in range(1, n_days):
        base_returns[i] += 0.05 * base_returns[i-1]  # Add some momentum
    
    # Add trend component
    trend = np.linspace(0, 0.3, n_days)  # 30% upward trend over period
    
    # Add volatility clustering (GARCH-like)
    volatility = np.zeros(n_days)
    volatility[0] = 0.02
    for i in range(1, n_days):
        volatility[i] = 0.00001 + 0.05 * base_returns[i-1]**2 + 0.9 * volatility[i-1]
        base_returns[i] *= np.sqrt(volatility[i])
    
    # Generate prices
    log_prices = np.cumsum(base_returns) + trend
    prices = 100 * np.exp(log_prices)
    
    # Generate OHLCV data
    high_prices = prices * (1 + np.abs(np.random.randn(n_days) * 0.01))
    low_prices = prices * (1 - np.abs(np.random.randn(n_days) * 0.01))
    open_prices = np.roll(prices, 1)
    open_prices[0] = 100
    volumes = np.random.randint(50000, 200000, n_days)
    
    # Create market data DataFrame
    market_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Add technical indicators manually (simulating data loader output)
    market_data['sma_20'] = market_data['close'].rolling(20).mean()
    market_data['sma_50'] = market_data['close'].rolling(50).mean()
    market_data['returns'] = market_data['close'].pct_change()
    market_data['volatility_20'] = market_data['returns'].rolling(20).std() * np.sqrt(252)
    
    print(f"Created market data: {len(market_data)} days")
    print(f"Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
    
    # ============================================================================
    # STEP 2: STRATEGY CREATION AND TESTING
    # ============================================================================
    print("\n2. CREATING AND TESTING TRADING STRATEGIES")
    print("-" * 50)
    
    # Create different strategies
    strategies = {
        'Momentum': MomentumAgent({
            'fast_period': 10,
            'slow_period': 30,
            'momentum_threshold': 0.02
        }),
        'Mean Reversion': MeanReversionAgent({
            'lookback': 20,
            'z_threshold': 1.5
        }),
        'Volatility Breakout': VolatilityBreakoutAgent({
            'vol_lookback': 20,
            'breakout_threshold': 2.0
        })
    }
    
    # Generate signals for each strategy
    strategy_signals = {}
    
    for name, strategy in strategies.items():
        print(f"Generating signals for {name} strategy...")
        
        if name == 'Momentum':
            signals_data = strategy.generate_detailed_signals(market_data)
            if signals_data is not None and 'signal' in signals_data.columns:
                strategy_signals[name] = signals_data['signal']
            else:
                # Fallback signal generation
                strategy_signals[name] = pd.Series(0, index=market_data.index)
        else:
            # Generate signals day by day for other strategies
            signals = []
            for i in range(len(market_data)):
                if i < 30:  # Need minimum data
                    signals.append(0)
                else:
                    data_slice = market_data.iloc[:i+1]
                    signal = strategy.generate_signal(data_slice)
                    signal_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
                    signals.append(signal_map.get(signal, 0))
            
            strategy_signals[name] = pd.Series(signals, index=market_data.index)
        
        signal_counts = strategy_signals[name].value_counts()
        print(f"  {name}: {signal_counts.to_dict()}")
    
    # ============================================================================
    # STEP 3: BACKTESTING WITH ADVANCED FEATURES
    # ============================================================================
    print("\n3. BACKTESTING STRATEGIES")
    print("-" * 50)
    
    # Configure backtesting
    backtest_config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        position_sizing='percent_risk',
        risk_per_trade=0.02
    )
    
    # Backtest each strategy
    backtest_results = {}
    
    for name, signals in strategy_signals.items():
        print(f"Backtesting {name} strategy...")
        
        backtester = EnhancedBacktester(market_data, backtest_config)
        results = backtester.backtest_strategy(signals)
        backtest_results[name] = results
        
        metrics = results['performance_metrics']
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Number of Trades: {metrics['num_trades']}")
    
    # ============================================================================
    # STEP 4: PORTFOLIO MANAGEMENT
    # ============================================================================
    print("\n4. PORTFOLIO MANAGEMENT")
    print("-" * 50)
    
    # Create portfolio manager
    portfolio_config = PortfolioConfig(
        initial_capital=300000,
        rebalance_frequency='monthly',
        risk_budget_method='equal_risk',
        max_strategy_weight=0.6
    )
    
    portfolio_manager = PortfolioManager(portfolio_config)
    
    # Add strategy returns to portfolio
    for name, results in backtest_results.items():
        returns = results['results_df']['returns']
        portfolio_manager.add_strategy(name, returns, name.lower().replace(' ', '_'))
    
    # Backtest portfolio
    print("Running portfolio backtest...")
    portfolio_results = portfolio_manager.backtest_portfolio()
    
    portfolio_metrics = portfolio_results['performance_metrics']
    print(f"Portfolio Results:")
    print(f"  Total Return: {portfolio_metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
    
    # ============================================================================
    # STEP 5: STRATEGY OPTIMIZATION
    # ============================================================================
    print("\n5. STRATEGY OPTIMIZATION")
    print("-" * 50)
    
    # Optimize the momentum strategy as an example
    def create_momentum_strategy(params):
        return MomentumAgent(params)
    
    def run_backtest(data, signals):
        backtester = EnhancedBacktester(data, backtest_config)
        return backtester.backtest_strategy(signals)
    
    # Define parameter space
    param_space = ParameterSpace()
    param_space.add_parameter('fast_period', 'integer', min=5, max=15)
    param_space.add_parameter('slow_period', 'integer', min=20, max=40)
    param_space.add_parameter('momentum_threshold', 'continuous', min=0.01, max=0.04)
    
    # Add constraint
    param_space.add_constraint(lambda p: p['fast_period'] < p['slow_period'])
    
    # Configure optimization
    opt_config = OptimizationConfig(
        method='grid_search',
        objective_metric='sharpe_ratio',
        max_iterations=20  # Keep small for demo
    )
    
    # Run optimization
    print("Running strategy optimization (limited iterations for demo)...")
    optimizer = StrategyOptimizer(opt_config)
    
    try:
        opt_results = optimizer.optimize_strategy(
            create_momentum_strategy, run_backtest, market_data, param_space
        )
        
        print(f"Optimization Results:")
        print(f"  Best Parameters: {opt_results['best_parameters']}")
        print(f"  Best Score: {opt_results['best_score']:.3f}")
    except Exception as e:
        print(f"Optimization failed: {e}")
        opt_results = None
    
    # ============================================================================
    # STEP 6: RISK ANALYSIS
    # ============================================================================
    print("\n6. RISK ANALYSIS")
    print("-" * 50)
    
    # Perform risk analysis on the best performing strategy
    best_strategy_name = max(backtest_results.keys(), 
                           key=lambda k: backtest_results[k]['performance_metrics']['sharpe_ratio'])
    best_results = backtest_results[best_strategy_name]
    best_returns = best_results['results_df']['returns']
    
    print(f"Analyzing risk for best strategy: {best_strategy_name}")
    
    # Configure risk analysis
    risk_config = RiskConfig(
        var_confidence_levels=[0.01, 0.05, 0.10],
        var_methods=['historical', 'parametric']
    )
    
    # Run risk analysis
    risk_analyzer = RiskAnalyzer(risk_config)
    risk_results = risk_analyzer.comprehensive_risk_analysis(
        best_returns, 
        portfolio_value=backtest_config.initial_capital
    )
    
    # Print key risk metrics
    basic_metrics = risk_results['basic_metrics']
    var_metrics = risk_results['var_metrics']
    
    print(f"Risk Analysis Results:")
    print(f"  Volatility: {basic_metrics['volatility']:.2%}")
    print(f"  Skewness: {basic_metrics['skewness']:.2f}")
    print(f"  Kurtosis: {basic_metrics['kurtosis']:.2f}")
    print(f"  VaR (5%): {var_metrics['5%']['var_historical']:.2%}")
    print(f"  CVaR (5%): {var_metrics['5%']['cvar_historical']:.2%}")
    
    # ============================================================================
    # STEP 7: VISUALIZATION
    # ============================================================================
    print("\n7. CREATING VISUALIZATIONS")
    print("-" * 50)
    
    # Create visualizer
    visualizer = TradingVisualizer()
    
    # Create comprehensive dashboard for best strategy
    print("Creating performance dashboard...")
    
    try:
        # Performance dashboard
        dashboard_fig = visualizer.plot_performance_dashboard(best_results)
        
        # Strategy comparison
        comparison_fig = visualizer.plot_strategy_comparison(backtest_results)
        
        # Interactive dashboard
        interactive_fig = visualizer.create_interactive_dashboard(
            best_results, best_strategy_name
        )
        
        print("Visualizations created successfully!")
        print("Note: In a Jupyter environment, these would display automatically.")
        print("To view in a script, add .show() to each figure.")
        
    except Exception as e:
        print(f"Visualization creation failed: {e}")
        print("This might be due to missing Plotly or display environment issues.")
    
    # ============================================================================
    # STEP 8: SUMMARY AND CONCLUSIONS
    # ============================================================================
    print("\n8. SUMMARY AND CONCLUSIONS")
    print("-" * 50)
    
    print("Trading System Analysis Complete!")
    print("\nKey Results:")
    
    # Best individual strategy
    best_individual = max(backtest_results.items(), 
                         key=lambda x: x[1]['performance_metrics']['sharpe_ratio'])
    print(f"  Best Individual Strategy: {best_individual[0]}")
    print(f"    Return: {best_individual[1]['performance_metrics']['total_return']:.2%}")
    print(f"    Sharpe: {best_individual[1]['performance_metrics']['sharpe_ratio']:.2f}")
    
    # Portfolio performance
    print(f"  Multi-Strategy Portfolio:")
    print(f"    Return: {portfolio_metrics['total_return']:.2%}")
    print(f"    Sharpe: {portfolio_metrics['sharpe_ratio']:.2f}")
    print(f"    Max Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
    
    # Risk assessment
    print(f"  Risk Assessment:")
    print(f"    Portfolio Volatility: {basic_metrics['volatility']:.2%}")
    print(f"    Tail Risk (VaR 5%): {var_metrics['5%']['var_historical']:.2%}")
    
    print("\nSystem Capabilities Demonstrated:")
    print("  ✓ Data loading and preprocessing")
    print("  ✓ Multiple trading strategies")
    print("  ✓ Advanced backtesting with transaction costs")
    print("  ✓ Portfolio management and optimization")
    print("  ✓ Strategy parameter optimization")
    print("  ✓ Comprehensive risk analysis")
    print("  ✓ Professional visualization tools")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    return {
        'market_data': market_data,
        'strategy_signals': strategy_signals,
        'backtest_results': backtest_results,
        'portfolio_results': portfolio_results,
        'optimization_results': opt_results,
        'risk_analysis': risk_results
    }


if __name__ == "__main__":
    # Run the complete demonstration
    results = main()
    
    # Additional analysis could be performed here
    print("\nAll results stored in 'results' dictionary for further analysis.")
    print("Available keys:", list(results.keys()))