"""
Comprehensive Visualization Tools for Trading Analysis

Advanced plotting and visualization utilities:
- Interactive charts with Plotly
- Performance dashboards
- Strategy comparison plots
- Risk visualization
- Portfolio analytics
- Technical analysis charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TradingVisualizer:
    """Comprehensive trading visualization toolkit"""
    
    def __init__(self, style: str = 'plotly_white'):
        self.style = style
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def plot_price_and_signals(self, data: pd.DataFrame, signals: pd.Series = None,
                              title: str = "Price Chart with Trading Signals",
                              figsize: Tuple[int, int] = (15, 8)) -> go.Figure:
        """Plot price chart with trading signals"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price and Signals', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Price',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if 'sma_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color=self.colors['secondary'], width=1)
                ),
                row=1, col=1
            )
        
        if 'sma_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color=self.colors['warning'], width=1)
                ),
                row=1, col=1
            )
        
        # Add trading signals
        if signals is not None:
            buy_signals = data[signals == 1]
            sell_signals = data[signals == -1]
            
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['close'],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color=self.colors['success']
                        )
                    ),
                    row=1, col=1
                )
            
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['close'],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color=self.colors['danger']
                        )
                    ),
                    row=1, col=1
                )
        
        # Volume chart
        if 'volume' in data.columns:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data['close'], data['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            template=self.style,
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def plot_candlestick_chart(self, data: pd.DataFrame, 
                             title: str = "Candlestick Chart",
                             indicators: List[str] = None) -> go.Figure:
        """Plot candlestick chart with technical indicators"""
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("Data must contain OHLC columns")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=('Price', 'Volume', 'Indicators'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color=self.colors['success'],
                decreasing_line_color=self.colors['danger']
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(128,128,128,0.5)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['bb_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(128,128,128,0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['bb_middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='rgba(128,128,128,0.7)', width=1)
                ),
                row=1, col=1
            )
        
        # Volume
        if 'volume' in data.columns:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data['close'], data['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Technical indicators
        if 'rsi_14' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['rsi_14'],
                    mode='lines',
                    name='RSI',
                    line=dict(color=self.colors['info'], width=2)
                ),
                row=3, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        fig.update_layout(
            title=title,
            template=self.style,
            height=800,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def plot_performance_dashboard(self, backtest_results: Dict,
                                 benchmark_data: pd.Series = None) -> go.Figure:
        """Create comprehensive performance dashboard"""
        results_df = backtest_results['results_df']
        metrics = backtest_results['performance_metrics']
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Portfolio Value', 'Drawdown', 'Rolling Sharpe',
                'Returns Distribution', 'Monthly Returns', 'Risk Metrics',
                'Cumulative Returns', 'Volatility', 'Trade Analysis'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Portfolio Value
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df['portfolio_value'],
                mode='lines',
                name='Portfolio',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        if benchmark_data is not None:
            benchmark_cumulative = (1 + benchmark_data).cumprod() * backtest_results['initial_capital']
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.colors['secondary'], width=2)
                ),
                row=1, col=1
            )
        
        # 2. Drawdown
        rolling_max = results_df['portfolio_value'].expanding().max()
        drawdown = (results_df['portfolio_value'] - rolling_max) / rolling_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                fill='tonexty',
                name='Drawdown',
                line=dict(color=self.colors['danger'], width=1),
                fillcolor='rgba(214, 39, 40, 0.3)'
            ),
            row=1, col=2
        )
        
        # 3. Rolling Sharpe
        rolling_returns = results_df['returns']
        rolling_sharpe = rolling_returns.rolling(60).mean() / rolling_returns.rolling(60).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=self.colors['info'], width=2)
            ),
            row=1, col=3
        )
        
        # 4. Returns Distribution
        fig.add_trace(
            go.Histogram(
                x=results_df['returns'] * 100,
                nbinsx=50,
                name='Returns',
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 5. Monthly Returns Heatmap
        monthly_returns = results_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot = monthly_returns.to_frame('returns')
        monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
        
        pivot_table = monthly_returns_pivot.pivot_table(
            values='returns', index='year', columns='month', fill_value=0
        ) * 100
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='RdYlGn',
                name='Monthly Returns',
                showscale=False
            ),
            row=2, col=2
        )
        
        # 6. Risk Metrics (Text)
        risk_text = f"""
        Sharpe: {metrics['sharpe_ratio']:.2f}<br>
        Sortino: {metrics['sortino_ratio']:.2f}<br>
        Max DD: {metrics['max_drawdown']:.2%}<br>
        Volatility: {metrics['volatility']:.2%}<br>
        VaR 95%: {metrics.get('var_95', 0):.2%}<br>
        Calmar: {metrics['calmar_ratio']:.2f}
        """
        
        fig.add_annotation(
            text=risk_text,
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12),
            row=2, col=3
        )
        
        # 7. Cumulative Returns
        cumulative_returns = (1 + results_df['returns']).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color=self.colors['success'], width=2)
            ),
            row=3, col=1
        )
        
        # 8. Rolling Volatility
        rolling_vol = results_df['returns'].rolling(30).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name='30D Volatility',
                line=dict(color=self.colors['warning'], width=2)
            ),
            row=3, col=2
        )
        
        # 9. Win/Loss Analysis
        if 'trades' in backtest_results:
            trades = backtest_results['trades']
            if trades:
                trade_pnls = [trade.get('pnl', 0) for trade in trades]
                wins = [pnl for pnl in trade_pnls if pnl > 0]
                losses = [pnl for pnl in trade_pnls if pnl < 0]
                
                fig.add_trace(
                    go.Bar(
                        x=['Wins', 'Losses'],
                        y=[len(wins), len(losses)],
                        name='Trade Count',
                        marker_color=[self.colors['success'], self.colors['danger']]
                    ),
                    row=3, col=3
                )
        
        fig.update_layout(
            title="Performance Dashboard",
            template=self.style,
            height=1200,
            showlegend=False
        )
        
        return fig
    
    def plot_strategy_comparison(self, strategies_results: Dict[str, Dict],
                               title: str = "Strategy Comparison") -> go.Figure:
        """Compare multiple strategies"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Portfolio Values', 'Risk-Return Scatter',
                'Drawdown Comparison', 'Performance Metrics'
            ]
        )
        
        # 1. Portfolio Values
        for name, results in strategies_results.items():
            results_df = results['results_df']
            fig.add_trace(
                go.Scatter(
                    x=results_df.index,
                    y=results_df['portfolio_value'],
                    mode='lines',
                    name=name,
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # 2. Risk-Return Scatter
        returns = []
        volatilities = []
        names = []
        
        for name, results in strategies_results.items():
            metrics = results['performance_metrics']
            returns.append(metrics['annualized_return'] * 100)
            volatilities.append(metrics['volatility'] * 100)
            names.append(name)
        
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers+text',
                text=names,
                textposition="top center",
                name='Strategies',
                marker=dict(size=10)
            ),
            row=1, col=2
        )
        
        # 3. Drawdown Comparison
        for name, results in strategies_results.items():
            results_df = results['results_df']
            rolling_max = results_df['portfolio_value'].expanding().max()
            drawdown = (results_df['portfolio_value'] - rolling_max) / rolling_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    mode='lines',
                    name=f'{name} DD',
                    line=dict(width=1)
                ),
                row=2, col=1
            )
        
        # 4. Performance Metrics Table
        metrics_data = []
        for name, results in strategies_results.items():
            metrics = results['performance_metrics']
            metrics_data.append([
                name,
                f"{metrics['total_return']:.2%}",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['max_drawdown']:.2%}",
                f"{metrics['volatility']:.2%}"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Strategy', 'Total Return', 'Sharpe', 'Max DD', 'Volatility'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*metrics_data)),
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            template=self.style,
            height=800
        )
        
        return fig
    
    def plot_correlation_analysis(self, returns_matrix: pd.DataFrame,
                                title: str = "Correlation Analysis") -> go.Figure:
        """Plot correlation analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Correlation Heatmap', 'Rolling Correlations',
                'PCA Analysis', 'Diversification Benefits'
            ]
        )
        
        # 1. Correlation Heatmap
        correlation_matrix = returns_matrix.corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                name='Correlation'
            ),
            row=1, col=1
        )
        
        # 2. Rolling Correlations (first two assets)
        if len(returns_matrix.columns) >= 2:
            asset1, asset2 = returns_matrix.columns[0], returns_matrix.columns[1]
            rolling_corr = returns_matrix[asset1].rolling(60).corr(returns_matrix[asset2])
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr,
                    mode='lines',
                    name=f'{asset1} vs {asset2}',
                    line=dict(width=2)
                ),
                row=1, col=2
            )
        
        # 3. PCA Analysis
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(returns_matrix.dropna())
        
        explained_variance = pca.explained_variance_ratio_[:10]  # First 10 components
        cumulative_variance = np.cumsum(explained_variance)
        
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(explained_variance) + 1)),
                y=explained_variance * 100,
                name='Individual',
                marker_color=self.colors['primary']
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_variance) + 1)),
                y=cumulative_variance * 100,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color=self.colors['danger'], width=2),
                yaxis='y2'
            ),
            row=2, col=1
        )
        
        # 4. Diversification Benefits
        equal_weight_portfolio = returns_matrix.mean(axis=1)
        individual_vol = returns_matrix.std() * np.sqrt(252) * 100
        portfolio_vol = equal_weight_portfolio.std() * np.sqrt(252) * 100
        
        diversification_ratio = individual_vol.mean() / portfolio_vol
        
        fig.add_trace(
            go.Bar(
                x=['Individual Assets (Avg)', 'Equal Weight Portfolio'],
                y=[individual_vol.mean(), portfolio_vol],
                name='Volatility',
                marker_color=[self.colors['warning'], self.colors['success']]
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            template=self.style,
            height=800
        )
        
        return fig
    
    def plot_factor_analysis(self, factor_results: Dict,
                           title: str = "Factor Analysis") -> go.Figure:
        """Plot factor analysis results"""
        if 'factor_loadings' not in factor_results:
            raise ValueError("Factor analysis results required")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Factor Loadings', 'Factor Contributions',
                'Risk Attribution', 'Factor Performance'
            ]
        )
        
        # 1. Factor Loadings
        factors = list(factor_results['factor_loadings'].keys())
        loadings = list(factor_results['factor_loadings'].values())
        
        fig.add_trace(
            go.Bar(
                x=factors,
                y=loadings,
                name='Factor Loadings',
                marker_color=self.colors['primary']
            ),
            row=1, col=1
        )
        
        # 2. Factor Contributions
        if 'factor_contributions' in factor_results:
            contributions = list(factor_results['factor_contributions'].values())
            
            fig.add_trace(
                go.Bar(
                    x=factors,
                    y=contributions,
                    name='Contributions',
                    marker_color=self.colors['success']
                ),
                row=1, col=2
            )
        
        # 3. Risk Attribution
        systematic_risk = factor_results.get('total_systematic_risk', 0)
        idiosyncratic_risk = factor_results.get('idiosyncratic_risk', 0)
        
        fig.add_trace(
            go.Pie(
                labels=['Systematic Risk', 'Idiosyncratic Risk'],
                values=[systematic_risk, idiosyncratic_risk],
                name='Risk Attribution'
            ),
            row=2, col=1
        )
        
        # 4. R-squared and Alpha
        r_squared = factor_results.get('r_squared', 0)
        alpha = factor_results.get('alpha', 0)
        
        fig.add_annotation(
            text=f"R-squared: {r_squared:.2%}<br>Alpha: {alpha:.2%}",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            template=self.style,
            height=600
        )
        
        return fig
    
    def create_interactive_dashboard(self, backtest_results: Dict,
                                   strategy_name: str = "Strategy") -> go.Figure:
        """Create comprehensive interactive dashboard"""
        results_df = backtest_results['results_df']
        metrics = backtest_results['performance_metrics']
        
        # Create main dashboard with multiple tabs
        fig = go.Figure()
        
        # Add portfolio value trace
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color=self.colors['primary'], width=3),
                hovertemplate='<b>Date</b>: %{x}<br>' +
                             '<b>Portfolio Value</b>: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            )
        )
        
        # Add benchmark line
        initial_value = backtest_results.get('initial_capital', 100000)
        fig.add_hline(
            y=initial_value,
            line_dash="dash",
            line_color="red",
            annotation_text="Initial Capital"
        )
        
        # Update layout with comprehensive styling
        fig.update_layout(
            title=dict(
                text=f"<b>{strategy_name} - Interactive Performance Dashboard</b>",
                x=0.5,
                font=dict(size=20)
            ),
            template=self.style,
            height=600,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            annotations=[
                dict(
                    text=f"Total Return: {metrics['total_return']:.2%} | " +
                         f"Sharpe: {metrics['sharpe_ratio']:.2f} | " +
                         f"Max DD: {metrics['max_drawdown']:.2%}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=1.02,
                    xanchor='center',
                    font=dict(size=12, color="gray")
                )
            ]
        )
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig


# Utility functions for quick plotting
def quick_performance_plot(returns: pd.Series, title: str = "Performance Analysis"):
    """Quick performance plot for returns series"""
    visualizer = TradingVisualizer()
    
    # Create simple performance data
    cumulative_returns = (1 + returns).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns,
            mode='lines',
            name='Cumulative Returns',
            line=dict(width=2)
        )
    )
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=400
    )
    
    return fig


def quick_drawdown_plot(portfolio_values: pd.Series, title: str = "Drawdown Analysis"):
    """Quick drawdown plot"""
    rolling_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - rolling_max) / rolling_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.3)'
        )
    )
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=300,
        yaxis_title="Drawdown (%)"
    )
    
    return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create sample OHLCV data
    base_price = 100
    returns = np.random.randn(len(dates)) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    high_prices = prices * (1 + np.abs(np.random.randn(len(dates)) * 0.01))
    low_prices = prices * (1 - np.abs(np.random.randn(len(dates)) * 0.01))
    open_prices = np.roll(prices, 1)
    open_prices[0] = base_price
    
    sample_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates)),
        'returns': np.concatenate([[0], np.diff(np.log(prices))])
    }, index=dates)
    
    # Add some technical indicators
    sample_data['sma_20'] = sample_data['close'].rolling(20).mean()
    sample_data['sma_50'] = sample_data['close'].rolling(50).mean()
    sample_data['rsi_14'] = 50 + 30 * np.sin(np.arange(len(dates)) * 0.1)  # Fake RSI
    
    # Create sample signals
    signals = pd.Series(0, index=dates)
    signals[sample_data['close'] > sample_data['sma_20']] = 1
    signals[sample_data['close'] < sample_data['sma_20']] = -1
    
    # Create visualizer
    visualizer = TradingVisualizer()
    
    # Example plots
    print("Creating visualization examples...")
    
    # 1. Price and signals chart
    price_fig = visualizer.plot_price_and_signals(sample_data, signals)
    price_fig.show()
    
    # 2. Candlestick chart
    candlestick_fig = visualizer.plot_candlestick_chart(sample_data)
    candlestick_fig.show()
    
    print("Visualization examples created successfully!")