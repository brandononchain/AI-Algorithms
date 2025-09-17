"""
Risk management system for the Quant Delta Market Maker.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from collections import deque

from data_models import (
    Portfolio, Position, Greeks, MarketData, RiskMetrics,
    MarketMakerConfig, OptionContract, Trade
)


@dataclass
class RiskLimit:
    """Risk limit definition."""
    name: str
    current_value: float
    limit_value: float
    warning_threshold: float = 0.8  # Warning at 80% of limit
    
    @property
    def utilization_pct(self) -> float:
        return abs(self.current_value) / self.limit_value * 100 if self.limit_value > 0 else 0
    
    @property
    def is_breached(self) -> bool:
        return abs(self.current_value) > self.limit_value
    
    @property
    def is_warning(self) -> bool:
        return abs(self.current_value) > (self.limit_value * self.warning_threshold)


class RiskManager:
    """
    Comprehensive risk management system.
    """
    
    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self.portfolio_history: deque = deque(maxlen=1440)  # 24 hours of minute data
        self.pnl_history: deque = deque(maxlen=1440)
        self.risk_events: List[Dict] = []
        self.last_risk_calculation = datetime.now()
        
        # Risk limits
        self.delta_limit = config.initial_capital * config.max_delta_exposure
        self.gamma_limit = config.initial_capital * config.max_gamma_exposure
        self.position_limit = config.max_position_size
        self.max_drawdown_limit = config.initial_capital * 0.1  # 10% max drawdown
        
        # VaR parameters
        self.var_confidence = 0.95
        self.var_lookback_days = 30
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_risk_metrics(
        self,
        portfolio: Portfolio,
        market_data: Dict[str, MarketData],
        greeks_by_symbol: Dict[str, Greeks]
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        
        # Portfolio Greeks
        portfolio_greeks = portfolio.get_portfolio_greeks(greeks_by_symbol)
        
        # Portfolio value and P&L
        portfolio_value = self._calculate_portfolio_value(portfolio, market_data)
        unrealized_pnl = portfolio_value - self.config.initial_capital
        
        # VaR calculation
        var_95 = self._calculate_var(portfolio, market_data)
        
        # Drawdown calculation
        max_drawdown = self._calculate_max_drawdown()
        
        # Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        risk_metrics = RiskMetrics(
            total_delta=portfolio_greeks.delta,
            total_gamma=portfolio_greeks.gamma,
            total_theta=portfolio_greeks.theta,
            total_vega=portfolio_greeks.vega,
            var_95=var_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio
        )
        
        # Store historical data
        self._update_history(portfolio_value, unrealized_pnl, risk_metrics)
        
        return risk_metrics
    
    def check_risk_limits(
        self,
        portfolio: Portfolio,
        market_data: Dict[str, MarketData],
        greeks_by_symbol: Dict[str, Greeks]
    ) -> List[RiskLimit]:
        """Check all risk limits and return violations."""
        
        risk_metrics = self.calculate_risk_metrics(portfolio, market_data, greeks_by_symbol)
        portfolio_value = self._calculate_portfolio_value(portfolio, market_data)
        
        # Define risk limits
        limits = [
            RiskLimit("Delta", risk_metrics.total_delta, self.delta_limit),
            RiskLimit("Gamma", risk_metrics.total_gamma, self.gamma_limit),
            RiskLimit("VaR 95%", risk_metrics.var_95, portfolio_value * 0.05),  # 5% of portfolio
            RiskLimit("Max Drawdown", risk_metrics.max_drawdown, self.max_drawdown_limit),
        ]
        
        # Check individual position limits
        for symbol, position in portfolio.positions.items():
            if position.contract:  # Options position
                limit = RiskLimit(
                    f"Position {symbol}",
                    abs(position.quantity),
                    self.position_limit
                )
                limits.append(limit)
        
        # Log violations
        violations = [limit for limit in limits if limit.is_breached]
        warnings = [limit for limit in limits if limit.is_warning and not limit.is_breached]
        
        for violation in violations:
            self.logger.error(f"RISK LIMIT BREACH: {violation.name} = {violation.current_value:.2f}, "
                            f"Limit = {violation.limit_value:.2f}")
            self._log_risk_event("LIMIT_BREACH", violation.name, violation.current_value)
        
        for warning in warnings:
            self.logger.warning(f"RISK WARNING: {warning.name} = {warning.current_value:.2f}, "
                              f"Limit = {warning.limit_value:.2f} ({warning.utilization_pct:.1f}%)")
        
        return limits
    
    def assess_trade_risk(
        self,
        proposed_trade: Dict[str, Any],
        portfolio: Portfolio,
        market_data: Dict[str, MarketData],
        greeks_by_symbol: Dict[str, Greeks]
    ) -> Tuple[bool, str]:
        """
        Assess risk of a proposed trade.
        
        Returns:
            (approved, reason)
        """
        symbol = proposed_trade['symbol']
        quantity = proposed_trade['quantity']
        side = proposed_trade['side']
        
        # Create hypothetical position change
        position_change = quantity if side == 'buy' else -quantity
        
        # Check position limits
        current_position = portfolio.get_position(symbol)
        current_qty = current_position.quantity if current_position else 0
        new_qty = current_qty + position_change
        
        if abs(new_qty) > self.position_limit:
            return False, f"Position limit exceeded: {abs(new_qty)} > {self.position_limit}"
        
        # Estimate impact on Greeks (simplified)
        if symbol in greeks_by_symbol:
            option_greeks = greeks_by_symbol[symbol]
            delta_impact = option_greeks.delta * position_change
            gamma_impact = option_greeks.gamma * position_change
            
            current_greeks = portfolio.get_portfolio_greeks(greeks_by_symbol)
            new_delta = current_greeks.delta + delta_impact
            new_gamma = current_greeks.gamma + gamma_impact
            
            if abs(new_delta) > self.delta_limit:
                return False, f"Delta limit would be exceeded: {abs(new_delta)} > {self.delta_limit}"
            
            if abs(new_gamma) > self.gamma_limit:
                return False, f"Gamma limit would be exceeded: {abs(new_gamma)} > {self.gamma_limit}"
        
        # Check concentration risk
        portfolio_value = self._calculate_portfolio_value(portfolio, market_data)
        trade_value = abs(proposed_trade.get('price', 0) * quantity)
        
        if trade_value > portfolio_value * 0.1:  # No single trade > 10% of portfolio
            return False, f"Trade size too large: {trade_value} > {portfolio_value * 0.1}"
        
        return True, "Trade approved"
    
    def calculate_position_sizing(
        self,
        symbol: str,
        target_delta: float,
        portfolio: Portfolio,
        market_data: Dict[str, MarketData],
        greeks_by_symbol: Dict[str, Greeks]
    ) -> int:
        """
        Calculate optimal position size based on risk constraints.
        """
        if symbol not in greeks_by_symbol:
            return 0
        
        option_greeks = greeks_by_symbol[symbol]
        if abs(option_greeks.delta) < 0.01:  # Avoid division by very small numbers
            return 0
        
        # Calculate position size to achieve target delta
        target_quantity = int(target_delta / option_greeks.delta)
        
        # Apply position limits
        target_quantity = max(-self.position_limit, min(target_quantity, self.position_limit))
        
        # Check current position
        current_position = portfolio.get_position(symbol)
        if current_position:
            max_change = self.position_limit - abs(current_position.quantity)
            if abs(target_quantity - current_position.quantity) > max_change:
                # Limit the change to stay within position limits
                if target_quantity > current_position.quantity:
                    target_quantity = current_position.quantity + max_change
                else:
                    target_quantity = current_position.quantity - max_change
        
        return target_quantity
    
    def get_risk_dashboard(
        self,
        portfolio: Portfolio,
        market_data: Dict[str, MarketData],
        greeks_by_symbol: Dict[str, Greeks]
    ) -> Dict[str, Any]:
        """Generate comprehensive risk dashboard."""
        
        risk_metrics = self.calculate_risk_metrics(portfolio, market_data, greeks_by_symbol)
        risk_limits = self.check_risk_limits(portfolio, market_data, greeks_by_symbol)
        portfolio_value = self._calculate_portfolio_value(portfolio, market_data)
        
        # Categorize limits
        breached = [l for l in risk_limits if l.is_breached]
        warnings = [l for l in risk_limits if l.is_warning and not l.is_breached]
        normal = [l for l in risk_limits if not l.is_warning]
        
        # Recent risk events
        recent_events = [e for e in self.risk_events if 
                        (datetime.now() - e['timestamp']).total_seconds() < 3600]
        
        return {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'risk_metrics': {
                'delta': risk_metrics.total_delta,
                'gamma': risk_metrics.total_gamma,
                'theta': risk_metrics.total_theta,
                'vega': risk_metrics.total_vega,
                'var_95': risk_metrics.var_95,
                'max_drawdown': risk_metrics.max_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio
            },
            'risk_limits': {
                'breached': len(breached),
                'warnings': len(warnings),
                'normal': len(normal),
                'details': {
                    'breached': [{'name': l.name, 'current': l.current_value, 'limit': l.limit_value} for l in breached],
                    'warnings': [{'name': l.name, 'current': l.current_value, 'limit': l.limit_value, 'utilization': l.utilization_pct} for l in warnings]
                }
            },
            'recent_events': len(recent_events),
            'positions': {
                'total': len(portfolio.positions),
                'options': len([p for p in portfolio.positions.values() if p.contract]),
                'underlying': len([p for p in portfolio.positions.values() if not p.contract])
            }
        }
    
    def _calculate_portfolio_value(
        self,
        portfolio: Portfolio,
        market_data: Dict[str, MarketData]
    ) -> float:
        """Calculate current portfolio value."""
        total_value = portfolio.cash
        
        for symbol, position in portfolio.positions.items():
            if symbol in market_data:
                market_price = market_data[symbol].mid_price
                position_value = position.quantity * market_price
                if position.contract:
                    position_value *= position.contract.multiplier
                total_value += position_value
        
        return total_value
    
    def _calculate_var(
        self,
        portfolio: Portfolio,
        market_data: Dict[str, MarketData]
    ) -> float:
        """Calculate Value at Risk using historical simulation."""
        if len(self.pnl_history) < 30:  # Need at least 30 data points
            return 0.0
        
        # Convert to numpy array for calculations
        pnl_array = np.array([p['unrealized_pnl'] for p in self.pnl_history])
        
        # Calculate daily returns
        if len(pnl_array) > 1:
            returns = np.diff(pnl_array) / pnl_array[:-1]
            returns = returns[~np.isnan(returns)]  # Remove NaN values
            
            if len(returns) > 0:
                # Calculate VaR at specified confidence level
                var_percentile = (1 - self.var_confidence) * 100
                var_value = np.percentile(returns, var_percentile)
                
                # Scale by current portfolio value
                current_value = self._calculate_portfolio_value(portfolio, market_data)
                return abs(var_value * current_value)
        
        return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak."""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        values = [p['portfolio_value'] for p in self.portfolio_history]
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = peak - value
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified)."""
        if len(self.pnl_history) < 30:
            return 0.0
        
        returns = [p['unrealized_pnl'] for p in self.pnl_history]
        returns_array = np.array(returns)
        
        if len(returns_array) > 1:
            daily_returns = np.diff(returns_array) / returns_array[:-1]
            daily_returns = daily_returns[~np.isnan(daily_returns)]
            
            if len(daily_returns) > 0 and np.std(daily_returns) > 0:
                excess_return = np.mean(daily_returns) - (self.config.risk_free_rate / 365)
                return excess_return / np.std(daily_returns) * np.sqrt(365)
        
        return 0.0
    
    def _update_history(self, portfolio_value: float, pnl: float, risk_metrics: RiskMetrics):
        """Update historical data."""
        timestamp = datetime.now()
        
        portfolio_snapshot = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'delta': risk_metrics.total_delta,
            'gamma': risk_metrics.total_gamma,
            'theta': risk_metrics.total_theta,
            'vega': risk_metrics.total_vega
        }
        
        pnl_snapshot = {
            'timestamp': timestamp,
            'unrealized_pnl': pnl,
            'var_95': risk_metrics.var_95,
            'max_drawdown': risk_metrics.max_drawdown
        }
        
        self.portfolio_history.append(portfolio_snapshot)
        self.pnl_history.append(pnl_snapshot)
    
    def _log_risk_event(self, event_type: str, description: str, value: float):
        """Log a risk event."""
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'description': description,
            'value': value
        }
        self.risk_events.append(event)
        
        # Keep only last 1000 events
        if len(self.risk_events) > 1000:
            self.risk_events = self.risk_events[-1000:]
    
    def emergency_liquidation_plan(
        self,
        portfolio: Portfolio,
        market_data: Dict[str, MarketData]
    ) -> List[Dict[str, Any]]:
        """
        Create emergency liquidation plan to reduce risk.
        """
        liquidation_orders = []
        
        # Sort positions by risk contribution
        positions_by_risk = []
        for symbol, position in portfolio.positions.items():
            if symbol in market_data:
                market_value = abs(position.quantity * market_data[symbol].mid_price)
                if position.contract:
                    market_value *= position.contract.multiplier
                
                positions_by_risk.append({
                    'symbol': symbol,
                    'position': position,
                    'market_value': market_value
                })
        
        # Sort by market value (largest first)
        positions_by_risk.sort(key=lambda x: x['market_value'], reverse=True)
        
        # Create liquidation orders for largest positions first
        for pos_info in positions_by_risk[:5]:  # Top 5 positions
            position = pos_info['position']
            symbol = pos_info['symbol']
            
            if symbol in market_data:
                market_data_item = market_data[symbol]
                
                liquidation_order = {
                    'symbol': symbol,
                    'quantity': abs(position.quantity),
                    'side': 'sell' if position.quantity > 0 else 'buy',
                    'price': market_data_item.bid if position.quantity > 0 else market_data_item.ask,
                    'urgency': 'emergency',
                    'reason': 'risk_reduction'
                }
                liquidation_orders.append(liquidation_order)
        
        return liquidation_orders