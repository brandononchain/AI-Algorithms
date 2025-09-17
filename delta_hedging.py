"""
Delta hedging and portfolio management for options market making.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from data_models import (
    Portfolio, Position, Order, OrderSide, OrderStatus, Trade,
    OptionContract, Greeks, MarketData, MarketMakerConfig
)
from greeks_calculator import GreeksCalculator


class DeltaHedger:
    """
    Manages delta hedging for an options portfolio.
    """
    
    def __init__(self, config: MarketMakerConfig, greeks_calculator: GreeksCalculator):
        self.config = config
        self.greeks_calc = greeks_calculator
        self.portfolio = Portfolio(cash=config.initial_capital)
        self.market_data: Dict[str, MarketData] = {}
        self.greeks_cache: Dict[str, Greeks] = {}
        self.last_hedge_time: Optional[datetime] = None
        self.hedge_history: List[Dict] = []
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def update_market_data(self, symbol: str, market_data: MarketData):
        """Update market data for a symbol."""
        self.market_data[symbol] = market_data
        
        # Update Greeks cache if this is an option
        position = self.portfolio.get_position(symbol)
        if position and position.contract:
            underlying_symbol = position.contract.underlying
            if underlying_symbol in self.market_data:
                underlying_price = self.market_data[underlying_symbol].mid_price
                
                greeks = self.greeks_calc.calculate_greeks(
                    underlying_price,
                    position.contract.strike,
                    position.contract.time_to_expiry,
                    market_data.implied_volatility or 0.2,
                    position.contract.option_type
                )
                self.greeks_cache[symbol] = greeks
    
    def calculate_portfolio_delta(self) -> float:
        """Calculate total portfolio delta."""
        total_delta = 0.0
        
        for symbol, position in self.portfolio.positions.items():
            if position.contract:  # Options position
                if symbol in self.greeks_cache:
                    option_delta = self.greeks_cache[symbol].delta * position.quantity
                    total_delta += option_delta
            else:  # Underlying position
                # Underlying has delta of 1
                total_delta += position.quantity
        
        return total_delta
    
    def calculate_hedge_requirement(self, target_delta: float = 0.0) -> Dict[str, int]:
        """
        Calculate required hedge trades to achieve target delta.
        
        Returns:
            Dictionary mapping underlying symbols to required position changes
        """
        current_delta = self.calculate_portfolio_delta()
        delta_to_hedge = current_delta - target_delta
        
        # For now, hedge everything with the primary underlying
        # In a multi-underlying portfolio, this would be more complex
        hedge_requirements = {}
        
        # Find the primary underlying (most common in options positions)
        underlying_symbols = set()
        for position in self.portfolio.positions.values():
            if position.contract:
                underlying_symbols.add(position.contract.underlying)
        
        if underlying_symbols:
            primary_underlying = list(underlying_symbols)[0]  # Simplification
            hedge_requirements[primary_underlying] = -int(delta_to_hedge)
        
        return hedge_requirements
    
    def should_hedge(self) -> bool:
        """Determine if hedging is needed based on thresholds."""
        current_delta = abs(self.calculate_portfolio_delta())
        delta_threshold = self.config.initial_capital * self.config.rebalance_threshold
        
        # Check delta threshold
        if current_delta > delta_threshold:
            return True
        
        # Check time-based hedging (hedge at least every hour)
        if self.last_hedge_time:
            time_since_hedge = datetime.now() - self.last_hedge_time
            if time_since_hedge > timedelta(hours=1):
                return True
        
        return False
    
    def execute_hedge_trades(self, hedge_requirements: Dict[str, int]) -> List[Order]:
        """
        Execute hedge trades.
        
        Returns:
            List of hedge orders placed
        """
        hedge_orders = []
        
        for symbol, quantity in hedge_requirements.items():
            if quantity == 0:
                continue
            
            if symbol not in self.market_data:
                self.logger.warning(f"No market data for {symbol}, skipping hedge")
                continue
            
            market_data = self.market_data[symbol]
            
            # Determine order side and price
            if quantity > 0:
                side = OrderSide.BUY
                price = market_data.ask  # Buy at ask
            else:
                side = OrderSide.SELL
                price = market_data.bid  # Sell at bid
                quantity = abs(quantity)
            
            # Create hedge order
            order_id = f"hedge_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                timestamp=datetime.now()
            )
            
            hedge_orders.append(order)
            self.portfolio.pending_orders[order_id] = order
            
            self.logger.info(f"Placed hedge order: {side.value} {quantity} {symbol} @ {price}")
        
        self.last_hedge_time = datetime.now()
        return hedge_orders
    
    def process_trade(self, trade: Trade):
        """Process a completed trade and update portfolio."""
        # Update position
        if trade.symbol in self.portfolio.positions:
            position = self.portfolio.positions[trade.symbol]
        else:
            position = Position(
                symbol=trade.symbol,
                quantity=0,
                avg_price=0,
                timestamp=trade.timestamp,
                contract=trade.contract
            )
            self.portfolio.positions[trade.symbol] = position
        
        # Update position based on trade
        trade_quantity = trade.quantity if trade.side == OrderSide.BUY else -trade.quantity
        position.update_position(trade_quantity, trade.price)
        
        # Update cash
        cash_flow = -trade.notional_value - trade.commission
        if trade.side == OrderSide.SELL:
            cash_flow = -cash_flow
        
        self.portfolio.cash += cash_flow
        
        # Remove from pending orders if it exists
        for order_id, order in list(self.portfolio.pending_orders.items()):
            if (order.symbol == trade.symbol and 
                order.side == trade.side and
                order.price == trade.price):
                order.filled_quantity += trade.quantity
                if order.is_filled:
                    del self.portfolio.pending_orders[order_id]
                break
        
        self.logger.info(f"Processed trade: {trade.side.value} {trade.quantity} {trade.symbol} @ {trade.price}")
    
    def run_hedging_cycle(self) -> Optional[List[Order]]:
        """
        Run a complete hedging cycle.
        
        Returns:
            List of hedge orders placed, or None if no hedging needed
        """
        if not self.should_hedge():
            return None
        
        current_delta = self.calculate_portfolio_delta()
        hedge_requirements = self.calculate_hedge_requirement()
        
        if not any(abs(qty) > 0 for qty in hedge_requirements.values()):
            return None
        
        # Log hedging decision
        hedge_info = {
            'timestamp': datetime.now(),
            'current_delta': current_delta,
            'hedge_requirements': hedge_requirements,
            'portfolio_value': self.get_portfolio_value()
        }
        self.hedge_history.append(hedge_info)
        
        self.logger.info(f"Hedging required - Current delta: {current_delta:.2f}")
        
        return self.execute_hedge_trades(hedge_requirements)
    
    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        total_value = self.portfolio.cash
        
        for symbol, position in self.portfolio.positions.items():
            if symbol in self.market_data:
                market_price = self.market_data[symbol].mid_price
                position_value = position.quantity * market_price
                if position.contract:
                    position_value *= position.contract.multiplier
                total_value += position_value
        
        return total_value
    
    def get_portfolio_greeks(self) -> Greeks:
        """Get total portfolio Greeks."""
        return self.portfolio.get_portfolio_greeks(self.greeks_cache)
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Calculate current risk metrics."""
        portfolio_greeks = self.get_portfolio_greeks()
        portfolio_value = self.get_portfolio_value()
        
        return {
            'portfolio_value': portfolio_value,
            'cash': self.portfolio.cash,
            'total_delta': portfolio_greeks.delta,
            'total_gamma': portfolio_greeks.gamma,
            'total_theta': portfolio_greeks.theta,
            'total_vega': portfolio_greeks.vega,
            'delta_pct': portfolio_greeks.delta / portfolio_value * 100 if portfolio_value > 0 else 0,
            'num_positions': len(self.portfolio.positions),
            'num_pending_orders': len(self.portfolio.pending_orders)
        }
    
    def gamma_scalp(self, symbol: str, price_history: List[float]) -> Optional[Order]:
        """
        Generate gamma scalping orders based on price movements.
        
        Args:
            symbol: Underlying symbol
            price_history: Recent price history
        
        Returns:
            Scalping order or None
        """
        if len(price_history) < 2:
            return None
        
        current_price = price_history[-1]
        previous_price = price_history[-2]
        
        # Get portfolio gamma for this underlying
        portfolio_gamma = 0.0
        for pos_symbol, position in self.portfolio.positions.items():
            if (position.contract and 
                position.contract.underlying == symbol and 
                pos_symbol in self.greeks_cache):
                portfolio_gamma += self.greeks_cache[pos_symbol].gamma * position.quantity
        
        if abs(portfolio_gamma) < 0.01:  # No significant gamma exposure
            return None
        
        signal = self.greeks_calc.gamma_scalp_signal(
            current_price, previous_price, portfolio_gamma
        )
        
        if not signal or symbol not in self.market_data:
            return None
        
        market_data = self.market_data[symbol]
        
        # Calculate scalp size based on gamma exposure
        scalp_size = min(int(abs(portfolio_gamma) * 100), 100)  # Limit scalp size
        
        if signal == 'buy':
            side = OrderSide.BUY
            price = market_data.ask
        else:
            side = OrderSide.SELL
            price = market_data.bid
        
        order_id = f"scalp_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=scalp_size,
            price=price,
            timestamp=datetime.now()
        )
        
        self.logger.info(f"Gamma scalp: {signal} {scalp_size} {symbol} @ {price}")
        return order