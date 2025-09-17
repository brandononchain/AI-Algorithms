"""
Real-time monitoring and dashboard for the Quant Delta Market Maker.
"""
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import logging
from typing import Dict, List, Any, Optional
import json

from data_models import MarketMakerConfig, Portfolio, Greeks
from market_maker import QuantDeltaMarketMaker
from risk_manager import RiskManager


class MarketMakerMonitor:
    """
    Real-time monitoring system for the market maker.
    """
    
    def __init__(self, market_maker: QuantDeltaMarketMaker, risk_manager: RiskManager):
        self.market_maker = market_maker
        self.risk_manager = risk_manager
        
        # Data storage for monitoring
        self.metrics_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.risk_alerts: List[Dict] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.update_interval = 5  # seconds
        
        # Dash app
        self.app = dash.Dash(__name__)
        self.setup_dashboard()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that collects metrics."""
        while self.is_monitoring:
            try:
                # Collect current metrics
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 data points
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Log key metrics
                if len(self.metrics_history) % 12 == 0:  # Every minute if 5s intervals
                    self.logger.info(
                        f"Portfolio: ${metrics['portfolio_value']:,.0f}, "
                        f"PnL: ${metrics['pnl']:+,.0f}, "
                        f"Delta: {metrics['delta']:+.2f}, "
                        f"Positions: {metrics['positions_count']}"
                    )
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current portfolio and risk metrics."""
        timestamp = datetime.now()
        
        # Get portfolio summary
        portfolio_summary = self.market_maker.get_portfolio_summary()
        
        # Get risk metrics
        risk_dashboard = self.risk_manager.get_risk_dashboard(
            self.market_maker.delta_hedger.portfolio,
            self.market_maker.market_data,
            self.market_maker.delta_hedger.greeks_cache
        )
        
        # Combine metrics
        metrics = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_summary['portfolio_value'],
            'cash': portfolio_summary['cash'],
            'pnl': portfolio_summary['pnl'],
            'pnl_pct': portfolio_summary['pnl_pct'],
            'delta': portfolio_summary['greeks']['delta'],
            'gamma': portfolio_summary['greeks']['gamma'],
            'theta': portfolio_summary['greeks']['theta'],
            'vega': portfolio_summary['greeks']['vega'],
            'positions_count': portfolio_summary['positions'],
            'total_trades': portfolio_summary['total_trades'],
            'runtime_hours': portfolio_summary['runtime_hours'],
            'risk_breaches': risk_dashboard['risk_limits']['breached'],
            'risk_warnings': risk_dashboard['risk_limits']['warnings'],
            'recent_risk_events': risk_dashboard['recent_events']
        }
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions."""
        alerts = []
        
        # P&L alerts
        if metrics['pnl'] < -50000:  # $50k loss
            alerts.append({
                'type': 'pnl_alert',
                'severity': 'high',
                'message': f"Large loss: ${metrics['pnl']:,.0f}",
                'timestamp': metrics['timestamp']
            })
        
        # Delta exposure alerts
        if abs(metrics['delta']) > 1000:
            alerts.append({
                'type': 'delta_alert',
                'severity': 'medium',
                'message': f"High delta exposure: {metrics['delta']:+.0f}",
                'timestamp': metrics['timestamp']
            })
        
        # Risk breach alerts
        if metrics['risk_breaches'] > 0:
            alerts.append({
                'type': 'risk_breach',
                'severity': 'high',
                'message': f"{metrics['risk_breaches']} risk limits breached",
                'timestamp': metrics['timestamp']
            })
        
        # Add alerts to history
        self.risk_alerts.extend(alerts)
        
        # Keep only last 100 alerts
        if len(self.risk_alerts) > 100:
            self.risk_alerts = self.risk_alerts[-100:]
        
        # Log high severity alerts
        for alert in alerts:
            if alert['severity'] == 'high':
                self.logger.warning(f"ALERT: {alert['message']}")
    
    def setup_dashboard(self):
        """Setup the Dash dashboard layout."""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Quant Delta Market Maker - Live Dashboard", 
                       style={'textAlign': 'center', 'marginBottom': 30}),
                html.Div(id='last-update', style={'textAlign': 'center', 'marginBottom': 20})
            ]),
            
            # Key metrics cards
            html.Div([
                html.Div([
                    html.H3("Portfolio Value", className="card-title"),
                    html.H2(id='portfolio-value', className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("P&L", className="card-title"),
                    html.H2(id='pnl-value', className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Delta", className="card-title"),
                    html.H2(id='delta-value', className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Positions", className="card-title"),
                    html.H2(id='positions-count', className="metric-value")
                ], className="metric-card")
            ], className="metrics-row"),
            
            # Charts
            html.Div([
                html.Div([
                    dcc.Graph(id='portfolio-chart')
                ], className="chart-container"),
                
                html.Div([
                    dcc.Graph(id='greeks-chart')
                ], className="chart-container")
            ], className="charts-row"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='pnl-chart')
                ], className="chart-container"),
                
                html.Div([
                    dcc.Graph(id='risk-gauge')
                ], className="chart-container")
            ], className="charts-row"),
            
            # Tables
            html.Div([
                html.Div([
                    html.H3("Recent Trades"),
                    dash_table.DataTable(
                        id='trades-table',
                        columns=[
                            {'name': 'Time', 'id': 'time'},
                            {'name': 'Symbol', 'id': 'symbol'},
                            {'name': 'Side', 'id': 'side'},
                            {'name': 'Quantity', 'id': 'quantity'},
                            {'name': 'Price', 'id': 'price'},
                            {'name': 'P&L', 'id': 'pnl'}
                        ],
                        style_cell={'textAlign': 'left'},
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{side} = buy'},
                                'backgroundColor': '#d4edda',
                            },
                            {
                                'if': {'filter_query': '{side} = sell'},
                                'backgroundColor': '#f8d7da',
                            }
                        ]
                    )
                ], className="table-container"),
                
                html.Div([
                    html.H3("Risk Alerts"),
                    dash_table.DataTable(
                        id='alerts-table',
                        columns=[
                            {'name': 'Time', 'id': 'time'},
                            {'name': 'Severity', 'id': 'severity'},
                            {'name': 'Message', 'id': 'message'}
                        ],
                        style_cell={'textAlign': 'left'},
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{severity} = high'},
                                'backgroundColor': '#f8d7da',
                                'color': 'black',
                            },
                            {
                                'if': {'filter_query': '{severity} = medium'},
                                'backgroundColor': '#fff3cd',
                                'color': 'black',
                            }
                        ]
                    )
                ], className="table-container")
            ], className="tables-row"),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
        
        # Setup callbacks
        self.setup_callbacks()
        
        # Add CSS
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    .metric-card {
                        background: white;
                        border-radius: 8px;
                        padding: 20px;
                        margin: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        text-align: center;
                        flex: 1;
                    }
                    .metrics-row {
                        display: flex;
                        justify-content: space-around;
                        margin-bottom: 30px;
                    }
                    .charts-row {
                        display: flex;
                        margin-bottom: 30px;
                    }
                    .chart-container {
                        flex: 1;
                        margin: 10px;
                    }
                    .tables-row {
                        display: flex;
                        margin-bottom: 30px;
                    }
                    .table-container {
                        flex: 1;
                        margin: 10px;
                    }
                    .metric-value {
                        color: #007bff;
                        margin: 10px 0;
                    }
                    body {
                        background-color: #f8f9fa;
                        font-family: 'Arial', sans-serif;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
    
    def setup_callbacks(self):
        """Setup Dash callbacks for real-time updates."""
        
        @self.app.callback(
            [
                Output('last-update', 'children'),
                Output('portfolio-value', 'children'),
                Output('pnl-value', 'children'),
                Output('delta-value', 'children'),
                Output('positions-count', 'children'),
                Output('portfolio-chart', 'figure'),
                Output('greeks-chart', 'figure'),
                Output('pnl-chart', 'figure'),
                Output('risk-gauge', 'figure'),
                Output('trades-table', 'data'),
                Output('alerts-table', 'data')
            ],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            if not self.metrics_history:
                # Return empty dashboard if no data
                empty_fig = go.Figure()
                return (
                    "No data available", "$0", "$0", "0", "0",
                    empty_fig, empty_fig, empty_fig, empty_fig,
                    [], []
                )
            
            latest = self.metrics_history[-1]
            
            # Format values
            last_update = f"Last Updated: {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            portfolio_val = f"${latest['portfolio_value']:,.0f}"
            pnl_val = f"${latest['pnl']:+,.0f} ({latest['pnl_pct']:+.2f}%)"
            delta_val = f"{latest['delta']:+.1f}"
            positions_val = str(latest['positions_count'])
            
            # Create charts
            portfolio_fig = self.create_portfolio_chart()
            greeks_fig = self.create_greeks_chart()
            pnl_fig = self.create_pnl_chart()
            risk_fig = self.create_risk_gauge()
            
            # Create tables
            trades_data = self.get_recent_trades_data()
            alerts_data = self.get_alerts_data()
            
            return (
                last_update, portfolio_val, pnl_val, delta_val, positions_val,
                portfolio_fig, greeks_fig, pnl_fig, risk_fig,
                trades_data, alerts_data
            )
    
    def create_portfolio_chart(self) -> go.Figure:
        """Create portfolio value chart."""
        if len(self.metrics_history) < 2:
            return go.Figure()
        
        df = pd.DataFrame(self.metrics_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Time',
            yaxis_title='Value ($)',
            hovermode='x unified'
        )
        
        return fig
    
    def create_greeks_chart(self) -> go.Figure:
        """Create Greeks chart."""
        if len(self.metrics_history) < 2:
            return go.Figure()
        
        df = pd.DataFrame(self.metrics_history)
        
        fig = go.Figure()
        
        # Delta
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['delta'],
            mode='lines',
            name='Delta',
            line=dict(color='red')
        ))
        
        # Gamma (scaled for visibility)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['gamma'] * 100,  # Scale gamma
            mode='lines',
            name='Gamma (×100)',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Portfolio Greeks',
            xaxis_title='Time',
            yaxis_title='Delta',
            yaxis2=dict(
                title='Gamma (×100)',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def create_pnl_chart(self) -> go.Figure:
        """Create P&L chart."""
        if len(self.metrics_history) < 2:
            return go.Figure()
        
        df = pd.DataFrame(self.metrics_history)
        
        # Color based on positive/negative
        colors = ['green' if pnl >= 0 else 'red' for pnl in df['pnl']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['pnl'],
            mode='lines+markers',
            name='P&L',
            line=dict(color='blue', width=2),
            marker=dict(color=colors, size=4)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='Profit & Loss',
            xaxis_title='Time',
            yaxis_title='P&L ($)',
            hovermode='x unified'
        )
        
        return fig
    
    def create_risk_gauge(self) -> go.Figure:
        """Create risk utilization gauge."""
        if not self.metrics_history:
            return go.Figure()
        
        latest = self.metrics_history[-1]
        
        # Calculate risk utilization (simplified)
        delta_util = min(abs(latest['delta']) / 1000 * 100, 100)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = delta_util,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Delta Risk Utilization (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        return fig
    
    def get_recent_trades_data(self) -> List[Dict]:
        """Get recent trades data for table."""
        # This would come from the market maker's trade history
        # For now, return sample data
        trades = self.market_maker.trades_executed[-10:]  # Last 10 trades
        
        trades_data = []
        for trade in trades:
            trades_data.append({
                'time': trade.timestamp.strftime('%H:%M:%S'),
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': f"${trade.price:.2f}",
                'pnl': f"${trade.notional_value - trade.commission:.2f}"
            })
        
        return trades_data
    
    def get_alerts_data(self) -> List[Dict]:
        """Get alerts data for table."""
        alerts_data = []
        for alert in self.risk_alerts[-20:]:  # Last 20 alerts
            alerts_data.append({
                'time': alert['timestamp'].strftime('%H:%M:%S'),
                'severity': alert['severity'],
                'message': alert['message']
            })
        
        return alerts_data
    
    def run_dashboard(self, host='127.0.0.1', port=8050, debug=False):
        """Run the dashboard server."""
        self.logger.info(f"Starting dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


def create_sample_monitor():
    """Create a sample monitor for demonstration."""
    from data_models import MarketMakerConfig
    
    # Create sample configuration
    config = MarketMakerConfig(
        initial_capital=1_000_000,
        max_delta_exposure=0.05,
        bid_ask_spread=0.02
    )
    
    # Create market maker and risk manager
    market_maker = QuantDeltaMarketMaker(config)
    risk_manager = RiskManager(config)
    
    # Create monitor
    monitor = MarketMakerMonitor(market_maker, risk_manager)
    
    return monitor


if __name__ == "__main__":
    # Run sample dashboard
    monitor = create_sample_monitor()
    monitor.start_monitoring()
    
    try:
        monitor.run_dashboard(debug=True)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("Dashboard stopped")