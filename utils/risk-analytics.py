"""
Comprehensive Risk Analytics and Performance Metrics

Advanced risk measurement and performance attribution:
- Value at Risk (VaR) and Conditional VaR
- Risk-adjusted returns
- Factor analysis and attribution
- Stress testing and scenario analysis  
- Risk budgeting and allocation
- Tail risk measures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
from datetime import datetime, timedelta
import yfinance as yf

warnings.filterwarnings('ignore')


@dataclass
class RiskConfig:
    """Configuration for risk analytics"""
    # VaR parameters
    var_confidence_levels: List[float] = None
    var_methods: List[str] = None  # 'historical', 'parametric', 'monte_carlo'
    
    # Stress testing
    stress_scenarios: Dict[str, float] = None
    monte_carlo_simulations: int = 10000
    
    # Factor analysis
    benchmark_symbols: List[str] = None
    factor_lookback: int = 252
    
    # Risk budgeting
    risk_budget_method: str = 'component_var'  # 'component_var', 'marginal_var'
    
    def __post_init__(self):
        if self.var_confidence_levels is None:
            self.var_confidence_levels = [0.01, 0.05, 0.10]
        
        if self.var_methods is None:
            self.var_methods = ['historical', 'parametric']
        
        if self.stress_scenarios is None:
            self.stress_scenarios = {
                'market_crash': -0.20,
                'moderate_decline': -0.10,
                'volatility_spike': 0.50,
                'interest_rate_shock': 0.02
            }
        
        if self.benchmark_symbols is None:
            self.benchmark_symbols = ['^GSPC', '^IXIC', '^RUT']  # S&P 500, NASDAQ, Russell 2000


class VaRCalculator:
    """Value at Risk calculations using different methods"""
    
    @staticmethod
    def historical_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate historical VaR"""
        return returns.quantile(confidence_level)
    
    @staticmethod
    def parametric_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate parametric VaR assuming normal distribution"""
        mu = returns.mean()
        sigma = returns.std()
        z_score = stats.norm.ppf(confidence_level)
        return mu + z_score * sigma
    
    @staticmethod
    def monte_carlo_var(returns: pd.Series, confidence_level: float = 0.05, 
                       n_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR"""
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random scenarios
        random_returns = np.random.normal(mu, sigma, n_simulations)
        
        return np.percentile(random_returns, confidence_level * 100)
    
    @staticmethod
    def conditional_var(returns: pd.Series, confidence_level: float = 0.05, 
                       method: str = 'historical') -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        if method == 'historical':
            var_threshold = VaRCalculator.historical_var(returns, confidence_level)
        elif method == 'parametric':
            var_threshold = VaRCalculator.parametric_var(returns, confidence_level)
        else:
            var_threshold = VaRCalculator.monte_carlo_var(returns, confidence_level)
        
        # Calculate expected value of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var_threshold
        
        return tail_returns.mean()


class RiskMetrics:
    """Comprehensive risk metrics calculation"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.var_calculator = VaRCalculator()
    
    def calculate_basic_metrics(self, returns: pd.Series, 
                              benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate basic risk and performance metrics"""
        metrics = {}
        
        # Return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Risk-adjusted returns
        risk_free_rate = 0.03  # 3% annual risk-free rate
        excess_returns = returns - risk_free_rate / 252
        metrics['sharpe_ratio'] = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Downside risk
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = excess_returns.mean() / downside_deviation * np.sqrt(252)
        else:
            metrics['sortino_ratio'] = np.inf
        
        # Drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        metrics['max_drawdown'] = drawdown.min()
        metrics['current_drawdown'] = drawdown.iloc[-1]
        
        # Calmar ratio
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = np.inf
        
        # Skewness and Kurtosis
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Win rate
        winning_periods = (returns > 0).sum()
        total_periods = len(returns)
        metrics['win_rate'] = winning_periods / total_periods if total_periods > 0 else 0
        
        # Average win/loss
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        if len(winning_returns) > 0:
            metrics['avg_win'] = winning_returns.mean()
        else:
            metrics['avg_win'] = 0
        
        if len(losing_returns) > 0:
            metrics['avg_loss'] = losing_returns.mean()
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else np.inf
        else:
            metrics['avg_loss'] = 0
            metrics['win_loss_ratio'] = np.inf
        
        # Benchmark comparison (if provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Beta
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            metrics['beta'] = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # Alpha
            benchmark_return = benchmark_returns.mean() * 252
            metrics['alpha'] = metrics['annualized_return'] - (risk_free_rate + metrics['beta'] * (benchmark_return - risk_free_rate))
            
            # Information ratio
            excess_returns_vs_benchmark = returns - benchmark_returns
            tracking_error = excess_returns_vs_benchmark.std() * np.sqrt(252)
            metrics['information_ratio'] = excess_returns_vs_benchmark.mean() / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
            
            # Correlation
            metrics['correlation'] = returns.corr(benchmark_returns)
        
        return metrics
    
    def calculate_var_metrics(self, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate VaR and CVaR for different confidence levels and methods"""
        var_metrics = {}
        
        for confidence_level in self.config.var_confidence_levels:
            var_metrics[f'{int(confidence_level * 100)}%'] = {}
            
            for method in self.config.var_methods:
                if method == 'historical':
                    var_value = self.var_calculator.historical_var(returns, confidence_level)
                elif method == 'parametric':
                    var_value = self.var_calculator.parametric_var(returns, confidence_level)
                elif method == 'monte_carlo':
                    var_value = self.var_calculator.monte_carlo_var(returns, confidence_level, 
                                                                  self.config.monte_carlo_simulations)
                else:
                    continue
                
                cvar_value = self.var_calculator.conditional_var(returns, confidence_level, method)
                
                var_metrics[f'{int(confidence_level * 100)}%'][f'var_{method}'] = var_value
                var_metrics[f'{int(confidence_level * 100)}%'][f'cvar_{method}'] = cvar_value
        
        return var_metrics
    
    def calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate tail risk metrics"""
        metrics = {}
        
        # Expected Shortfall at different levels
        for confidence_level in [0.01, 0.05, 0.10]:
            var_threshold = self.var_calculator.historical_var(returns, confidence_level)
            tail_returns = returns[returns <= var_threshold]
            
            if len(tail_returns) > 0:
                metrics[f'expected_shortfall_{int(confidence_level * 100)}%'] = tail_returns.mean()
            else:
                metrics[f'expected_shortfall_{int(confidence_level * 100)}%'] = var_threshold
        
        # Tail ratio
        right_tail = returns.quantile(0.95)
        left_tail = returns.quantile(0.05)
        metrics['tail_ratio'] = abs(right_tail / left_tail) if left_tail != 0 else np.inf
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        metrics['max_consecutive_losses'] = max_consecutive_losses
        
        return metrics
    
    def stress_test(self, portfolio_value: float, returns: pd.Series, 
                   positions: Dict[str, float] = None) -> Dict[str, float]:
        """Perform stress testing under various scenarios"""
        stress_results = {}
        
        for scenario_name, shock_magnitude in self.config.stress_scenarios.items():
            if scenario_name == 'market_crash' or scenario_name == 'moderate_decline':
                # Apply negative shock to returns
                stressed_return = shock_magnitude
                stressed_portfolio_value = portfolio_value * (1 + stressed_return)
                stress_results[scenario_name] = {
                    'portfolio_value': stressed_portfolio_value,
                    'loss': portfolio_value - stressed_portfolio_value,
                    'loss_percentage': stressed_return
                }
                
            elif scenario_name == 'volatility_spike':
                # Calculate impact of volatility increase
                current_vol = returns.std() * np.sqrt(252)
                stressed_vol = current_vol * (1 + shock_magnitude)
                
                # Estimate VaR under stressed volatility
                stressed_var = returns.mean() + stats.norm.ppf(0.05) * (stressed_vol / np.sqrt(252))
                stressed_portfolio_value = portfolio_value * (1 + stressed_var)
                
                stress_results[scenario_name] = {
                    'portfolio_value': stressed_portfolio_value,
                    'loss': portfolio_value - stressed_portfolio_value,
                    'loss_percentage': stressed_var,
                    'stressed_volatility': stressed_vol
                }
        
        return stress_results
    
    def calculate_risk_contribution(self, returns_matrix: pd.DataFrame, 
                                  weights: np.ndarray) -> Dict[str, Any]:
        """Calculate risk contribution of each component"""
        # Calculate portfolio return
        portfolio_returns = (returns_matrix * weights).sum(axis=1)
        portfolio_var = portfolio_returns.var()
        
        # Calculate marginal VaR
        marginal_var = {}
        component_var = {}
        
        for i, asset in enumerate(returns_matrix.columns):
            # Marginal VaR: derivative of portfolio variance with respect to weight
            marginal_var[asset] = 2 * weights[i] * returns_matrix[asset].cov(portfolio_returns) / portfolio_var if portfolio_var > 0 else 0
            
            # Component VaR: weight * marginal VaR
            component_var[asset] = weights[i] * marginal_var[asset]
        
        # Risk contribution as percentage
        total_component_var = sum(component_var.values())
        risk_contribution_pct = {
            asset: component_var[asset] / total_component_var * 100 if total_component_var != 0 else 0
            for asset in component_var
        }
        
        return {
            'marginal_var': marginal_var,
            'component_var': component_var,
            'risk_contribution_pct': risk_contribution_pct,
            'portfolio_var': portfolio_var
        }


class FactorAnalysis:
    """Factor analysis and performance attribution"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.factor_data = None
        self.factor_loadings = None
    
    def load_factor_data(self, start_date: str = None, end_date: str = None):
        """Load factor data (market benchmarks)"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=self.config.factor_lookback * 2)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        factor_data = {}
        
        for symbol in self.config.benchmark_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    returns = data['Close'].pct_change().dropna()
                    factor_data[symbol] = returns
            except Exception as e:
                print(f"Error loading factor data for {symbol}: {e}")
        
        if factor_data:
            self.factor_data = pd.DataFrame(factor_data)
            self.factor_data = self.factor_data.dropna()
    
    def perform_factor_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """Perform factor analysis using regression"""
        if self.factor_data is None:
            self.load_factor_data()
        
        if self.factor_data is None or self.factor_data.empty:
            return {'error': 'No factor data available'}
        
        # Align dates
        common_dates = returns.index.intersection(self.factor_data.index)
        if len(common_dates) < 60:  # Minimum data requirement
            return {'error': 'Insufficient overlapping data'}
        
        returns_aligned = returns[common_dates]
        factors_aligned = self.factor_data.loc[common_dates]
        
        # Multiple regression
        X = factors_aligned.values
        y = returns_aligned.values
        
        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # Fit regression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_with_const, y)
        
        # Extract results
        alpha = reg.coef_[0] * 252  # Annualized alpha
        factor_loadings = dict(zip(self.factor_data.columns, reg.coef_[1:]))
        
        # Calculate R-squared
        y_pred = reg.predict(X_with_const)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Factor contribution to return
        factor_contributions = {}
        for factor, loading in factor_loadings.items():
            factor_return = factors_aligned[factor].mean() * 252
            factor_contributions[factor] = loading * factor_return
        
        # Residual risk
        residuals = y - y_pred
        idiosyncratic_risk = np.std(residuals) * np.sqrt(252)
        
        return {
            'alpha': alpha,
            'factor_loadings': factor_loadings,
            'factor_contributions': factor_contributions,
            'r_squared': r_squared,
            'idiosyncratic_risk': idiosyncratic_risk,
            'total_systematic_risk': np.sqrt(np.var(y_pred)) * np.sqrt(252)
        }
    
    def perform_pca_analysis(self, returns_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform Principal Component Analysis"""
        # Standardize returns
        returns_std = (returns_matrix - returns_matrix.mean()) / returns_matrix.std()
        returns_std = returns_std.dropna()
        
        # Perform PCA
        pca = PCA()
        pca.fit(returns_std)
        
        # Extract results
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Principal components
        components = pd.DataFrame(
            pca.components_[:5],  # First 5 components
            columns=returns_matrix.columns,
            index=[f'PC{i+1}' for i in range(5)]
        )
        
        # Transform data
        transformed_data = pca.transform(returns_std)
        
        return {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'components': components,
            'n_components_90_variance': np.argmax(cumulative_variance >= 0.9) + 1,
            'transformed_data': transformed_data
        }


class RiskAnalyzer:
    """Main risk analysis class"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.risk_metrics = RiskMetrics(config)
        self.factor_analysis = FactorAnalysis(config)
    
    def comprehensive_risk_analysis(self, returns: pd.Series, 
                                  benchmark_returns: pd.Series = None,
                                  portfolio_value: float = 100000) -> Dict[str, Any]:
        """Perform comprehensive risk analysis"""
        analysis_results = {}
        
        # Basic metrics
        analysis_results['basic_metrics'] = self.risk_metrics.calculate_basic_metrics(
            returns, benchmark_returns
        )
        
        # VaR metrics
        analysis_results['var_metrics'] = self.risk_metrics.calculate_var_metrics(returns)
        
        # Tail risk metrics
        analysis_results['tail_risk'] = self.risk_metrics.calculate_tail_risk_metrics(returns)
        
        # Stress testing
        analysis_results['stress_test'] = self.risk_metrics.stress_test(
            portfolio_value, returns
        )
        
        # Factor analysis
        analysis_results['factor_analysis'] = self.factor_analysis.perform_factor_analysis(returns)
        
        return analysis_results
    
    def plot_risk_analysis(self, results: Dict[str, Any], returns: pd.Series,
                          figsize: Tuple[int, int] = (16, 12)):
        """Plot comprehensive risk analysis"""
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Return distribution
        axes[0].hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(returns.mean() * 100, color='red', linestyle='--', 
                       label=f'Mean: {returns.mean()*100:.2f}%')
        axes[0].set_title('Return Distribution')
        axes[0].set_xlabel('Daily Return (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        axes[1].plot(cumulative_returns.index, cumulative_returns, linewidth=2)
        axes[1].set_title('Cumulative Returns')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        axes[2].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        axes[2].plot(drawdown.index, drawdown, color='red', linewidth=1)
        axes[2].set_title(f'Drawdown (Max: {results["basic_metrics"]["max_drawdown"]:.2%})')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Rolling volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        axes[3].plot(rolling_vol.index, rolling_vol, linewidth=1)
        axes[3].set_title('30-Day Rolling Volatility')
        axes[3].set_ylabel('Volatility (%)')
        axes[3].grid(True, alpha=0.3)
        
        # 5. VaR comparison
        if 'var_metrics' in results:
            var_data = results['var_metrics']
            confidence_levels = list(var_data.keys())
            historical_vars = [var_data[level]['var_historical'] * 100 for level in confidence_levels]
            parametric_vars = [var_data[level]['var_parametric'] * 100 for level in confidence_levels]
            
            x = np.arange(len(confidence_levels))
            width = 0.35
            
            axes[4].bar(x - width/2, historical_vars, width, label='Historical VaR', alpha=0.7)
            axes[4].bar(x + width/2, parametric_vars, width, label='Parametric VaR', alpha=0.7)
            axes[4].set_title('VaR Comparison')
            axes[4].set_xlabel('Confidence Level')
            axes[4].set_ylabel('VaR (%)')
            axes[4].set_xticks(x)
            axes[4].set_xticklabels(confidence_levels)
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
        
        # 6. Q-Q plot
        from scipy.stats import probplot
        probplot(returns, dist="norm", plot=axes[5])
        axes[5].set_title('Q-Q Plot (Normal Distribution)')
        axes[5].grid(True, alpha=0.3)
        
        # 7. Factor loadings (if available)
        if 'factor_analysis' in results and 'factor_loadings' in results['factor_analysis']:
            factor_loadings = results['factor_analysis']['factor_loadings']
            factors = list(factor_loadings.keys())
            loadings = list(factor_loadings.values())
            
            axes[6].bar(factors, loadings, alpha=0.7)
            axes[6].set_title('Factor Loadings')
            axes[6].set_ylabel('Loading')
            axes[6].tick_params(axis='x', rotation=45)
            axes[6].grid(True, alpha=0.3)
        
        # 8. Stress test results
        if 'stress_test' in results:
            stress_data = results['stress_test']
            scenarios = list(stress_data.keys())
            losses = [stress_data[scenario]['loss_percentage'] * 100 for scenario in scenarios]
            
            axes[7].bar(scenarios, losses, alpha=0.7, color='red')
            axes[7].set_title('Stress Test Results')
            axes[7].set_ylabel('Loss (%)')
            axes[7].tick_params(axis='x', rotation=45)
            axes[7].grid(True, alpha=0.3)
        
        # 9. Risk metrics summary (text)
        basic_metrics = results['basic_metrics']
        summary_text = f"""
        Sharpe Ratio: {basic_metrics['sharpe_ratio']:.2f}
        Sortino Ratio: {basic_metrics['sortino_ratio']:.2f}
        Max Drawdown: {basic_metrics['max_drawdown']:.2%}
        Volatility: {basic_metrics['volatility']:.2%}
        Skewness: {basic_metrics['skewness']:.2f}
        Kurtosis: {basic_metrics['kurtosis']:.2f}
        Win Rate: {basic_metrics['win_rate']:.2%}
        """
        
        axes[8].text(0.1, 0.9, summary_text, transform=axes[8].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[8].set_title('Risk Metrics Summary')
        axes[8].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        self._print_risk_summary(results)
    
    def _print_risk_summary(self, results: Dict[str, Any]):
        """Print formatted risk analysis summary"""
        print("=" * 80)
        print("COMPREHENSIVE RISK ANALYSIS REPORT")
        print("=" * 80)
        
        # Basic metrics
        basic = results['basic_metrics']
        print("\nPERFORMANCE METRICS:")
        print("-" * 40)
        print(f"Total Return:         {basic['total_return']:.2%}")
        print(f"Annualized Return:    {basic['annualized_return']:.2%}")
        print(f"Volatility:           {basic['volatility']:.2%}")
        print(f"Sharpe Ratio:         {basic['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:        {basic['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:         {basic['calmar_ratio']:.2f}")
        
        # Risk metrics
        print("\nRISK METRICS:")
        print("-" * 40)
        print(f"Maximum Drawdown:     {basic['max_drawdown']:.2%}")
        print(f"Current Drawdown:     {basic['current_drawdown']:.2%}")
        print(f"Skewness:             {basic['skewness']:.2f}")
        print(f"Kurtosis:             {basic['kurtosis']:.2f}")
        print(f"Win Rate:             {basic['win_rate']:.2%}")
        print(f"Win/Loss Ratio:       {basic['win_loss_ratio']:.2f}")
        
        # VaR metrics
        if 'var_metrics' in results:
            print("\nVALUE AT RISK:")
            print("-" * 40)
            for level, metrics in results['var_metrics'].items():
                print(f"{level} VaR (Historical):  {metrics['var_historical']:.2%}")
                print(f"{level} CVaR (Historical): {metrics['cvar_historical']:.2%}")
        
        # Factor analysis
        if 'factor_analysis' in results and 'alpha' in results['factor_analysis']:
            fa = results['factor_analysis']
            print("\nFACTOR ANALYSIS:")
            print("-" * 40)
            print(f"Alpha (Annualized):   {fa['alpha']:.2%}")
            print(f"R-squared:            {fa['r_squared']:.2%}")
            print(f"Idiosyncratic Risk:   {fa['idiosyncratic_risk']:.2%}")
            
            print("\nFactor Loadings:")
            for factor, loading in fa['factor_loadings'].items():
                print(f"  {factor}: {loading:.3f}")
        
        print("=" * 80)


# Example usage
if __name__ == "__main__":
    # Generate sample return data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create returns with some realistic characteristics
    base_returns = np.random.randn(len(dates)) * 0.01
    volatility_clustering = np.random.randn(len(dates)) * 0.005
    trend = np.linspace(0, 0.0002, len(dates))  # Slight upward trend
    
    returns = base_returns + volatility_clustering + trend
    returns = pd.Series(returns, index=dates)
    
    # Create risk analyzer
    config = RiskConfig()
    analyzer = RiskAnalyzer(config)
    
    # Perform comprehensive analysis
    print("Running comprehensive risk analysis...")
    results = analyzer.comprehensive_risk_analysis(returns, portfolio_value=1000000)
    
    # Plot results
    analyzer.plot_risk_analysis(results, returns)