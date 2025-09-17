"""
Options market maker with delta hedging capabilities.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
import asyncio
from collections import defaultdict
import uuid

from data_models import (
    OptionContract, MarketData, Order, OrderSide, OrderStatus, Trade,
    MarketMakerConfig, Greeks, OptionType
)
from greeks_calculator import GreeksCalculator
from delta_hedging import DeltaHedger


class QuantDeltaMarketMaker:
    """
    Sophisticated options market maker with delta hedging.
    """
    
    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self.greeks_calc = GreeksCalculator(config.risk_free_rate)
        self.delta_hedger = DeltaHedger(config, self.greeks_calc)
        
        # Market data and contracts
        self.market_data: Dict[str, MarketData] = {}
        self.option_contracts: Dict[str, OptionContract] = {}
        self.underlying_symbols: Set[str] = set()
        
        # Quote management
        self.active_quotes: Dict[str, List[Order]] = defaultdict(list)
        self.quote_history: List[Dict] = []
        self.last_quote_time: Dict[str, datetime] = {}
        
        # Performance tracking
        self.trades_executed: List[Trade] = []
        self.pnl_history: List[Dict] = []
        self.start_time = datetime.now()
        
        # State
        self.is_running = False
        self.last_risk_check = datetime.now()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_option_contract(self, contract: OptionContract):
        """Add an option contract to trade."""
        self.option_contracts[contract.contract_id] = contract
        self.underlying_symbols.add(contract.underlying)
        self.logger.info(f"Added contract: {contract.contract_id}")
    
    def update_market_data(self, symbol: str, market_data: MarketData):
        """Update market data for a symbol."""
        self.market_data[symbol] = market_data
        self.delta_hedger.update_market_data(symbol, market_data)
    
    def calculate_fair_value(self, contract: OptionContract) -> Optional[float]:
        """Calculate fair value for an option contract."""
        underlying_symbol = contract.underlying
        
        if underlying_symbol not in self.market_data:
            return None
        
        underlying_price = self.market_data[underlying_symbol].mid_price
        
        # Use implied volatility if available, otherwise estimate
        if contract.contract_id in self.market_data:
            option_data = self.market_data[contract.contract_id]
            volatility = option_data.implied_volatility or 0.2
        else:
            # Estimate volatility (this would be more sophisticated in practice)
            volatility = 0.2
        
        fair_value = self.greeks_calc.black_scholes_price(
            underlying_price,
            contract.strike,
            contract.time_to_expiry,
            volatility,
            contract.option_type
        )
        
        return fair_value
    
    def calculate_bid_ask_spread(self, contract: OptionContract, fair_value: float) -> Tuple[float, float]:
        """
        Calculate bid and ask prices based on fair value and market conditions.
        """
        base_spread = self.config.bid_ask_spread
        
        # Adjust spread based on time to expiry
        time_adjustment = min(1.0 / max(contract.time_to_expiry, 0.01), 2.0)
        
        # Adjust spread based on volatility
        underlying_symbol = contract.underlying
        if underlying_symbol in self.market_data:
            underlying_data = self.market_data[underlying_symbol]
            vol_adjustment = 1 + underlying_data.spread_pct
        else:
            vol_adjustment = 1.0
        
        # Adjust spread based on current position
        position_adjustment = self._get_position_adjustment(contract)
        
        # Calculate final spread
        adjusted_spread = base_spread * time_adjustment * vol_adjustment * position_adjustment
        half_spread = fair_value * adjusted_spread / 2
        
        bid = fair_value - half_spread
        ask = fair_value + half_spread
        
        # Ensure minimum tick size and positive prices
        bid = max(round(bid, 2), 0.01)
        ask = max(round(ask, 2), bid + 0.01)
        
        return bid, ask
    
    def _get_position_adjustment(self, contract: OptionContract) -> float:
        """
        Adjust spread based on current position in the contract.
        Wider spreads when we have large positions to encourage unwinding.
        """
        position = self.delta_hedger.portfolio.get_position(contract.contract_id)
        if not position:
            return 1.0
        
        # Calculate position as percentage of max position size
        position_pct = abs(position.quantity) / self.config.max_position_size
        
        # Increase spread by up to 50% for large positions
        return 1.0 + (position_pct * 0.5)
    
    def generate_quotes(self, contract: OptionContract) -> Optional[Tuple[Order, Order]]:
        """Generate bid and ask quotes for an option contract."""
        fair_value = self.calculate_fair_value(contract)
        if fair_value is None:
            return None
        
        bid_price, ask_price = self.calculate_bid_ask_spread(contract, fair_value)
        
        # Check if we should quote (position limits, risk limits, etc.)
        if not self._should_quote(contract):
            return None
        
        # Determine quote sizes
        bid_size, ask_size = self._calculate_quote_sizes(contract)
        
        # Create orders
        bid_order = Order(
            order_id=f"bid_{contract.contract_id}_{uuid.uuid4().hex[:8]}",
            symbol=contract.contract_id,
            side=OrderSide.BUY,
            quantity=bid_size,
            price=bid_price,
            timestamp=datetime.now(),
            contract=contract
        )
        
        ask_order = Order(
            order_id=f"ask_{contract.contract_id}_{uuid.uuid4().hex[:8]}",
            symbol=contract.contract_id,
            side=OrderSide.SELL,
            quantity=ask_size,
            price=ask_price,
            timestamp=datetime.now(),
            contract=contract
        )
        
        return bid_order, ask_order
    
    def _should_quote(self, contract: OptionContract) -> bool:
        """Determine if we should provide quotes for this contract."""
        # Check if we have recent market data
        if contract.underlying not in self.market_data:
            return False
        
        underlying_data = self.market_data[contract.underlying]
        if (datetime.now() - underlying_data.timestamp).total_seconds() > 60:
            return False
        
        # Check position limits
        position = self.delta_hedger.portfolio.get_position(contract.contract_id)
        if position and abs(position.quantity) >= self.config.max_position_size:
            return False
        
        # Check if we have too many pending orders
        pending_count = len(self.active_quotes[contract.contract_id])
        if pending_count >= self.config.max_orders_per_symbol:
            return False
        
        # Check time to expiry (don't quote very close to expiry)
        if contract.time_to_expiry < 1/365:  # Less than 1 day
            return False
        
        return True
    
    def _calculate_quote_sizes(self, contract: OptionContract) -> Tuple[int, int]:
        """Calculate bid and ask sizes for quotes."""
        base_size = 10  # Base quote size
        
        # Adjust based on time to expiry
        if contract.time_to_expiry > 0.25:  # More than 3 months
            size_multiplier = 2
        elif contract.time_to_expiry > 0.08:  # More than 1 month
            size_multiplier = 1.5
        else:
            size_multiplier = 1
        
        # Adjust based on current position
        position = self.delta_hedger.portfolio.get_position(contract.contract_id)
        if position:
            if position.quantity > 0:  # Long position - prefer to sell
                bid_size = int(base_size * size_multiplier * 0.5)
                ask_size = int(base_size * size_multiplier * 1.5)
            else:  # Short position - prefer to buy
                bid_size = int(base_size * size_multiplier * 1.5)
                ask_size = int(base_size * size_multiplier * 0.5)
        else:
            bid_size = ask_size = int(base_size * size_multiplier)
        
        return max(bid_size, 1), max(ask_size, 1)
    
    def update_quotes(self):
        """Update quotes for all option contracts."""
        current_time = datetime.now()
        
        for contract_id, contract in self.option_contracts.items():
            # Check if we need to refresh quotes
            last_quote = self.last_quote_time.get(contract_id)
            if (last_quote and 
                (current_time - last_quote).total_seconds() < self.config.quote_refresh_interval):
                continue
            
            # Cancel existing quotes
            self._cancel_quotes(contract_id)
            
            # Generate new quotes
            quotes = self.generate_quotes(contract)
            if quotes:
                bid_order, ask_order = quotes
                self.active_quotes[contract_id] = [bid_order, ask_order]
                self.last_quote_time[contract_id] = current_time
                
                # Log quote update
                self.logger.info(
                    f"Updated quotes for {contract_id}: "
                    f"Bid {bid_order.quantity}@{bid_order.price} "
                    f"Ask {ask_order.quantity}@{ask_order.price}"
                )
    
    def _cancel_quotes(self, contract_id: str):
        """Cancel existing quotes for a contract."""
        if contract_id in self.active_quotes:
            for order in self.active_quotes[contract_id]:
                order.status = OrderStatus.CANCELLED
            self.active_quotes[contract_id].clear()
    
    def process_trade_execution(self, trade: Trade):
        """Process a trade execution."""
        self.trades_executed.append(trade)
        self.delta_hedger.process_trade(trade)
        
        # Cancel the filled order
        contract_id = trade.contract.contract_id if trade.contract else trade.symbol
        if contract_id in self.active_quotes:
            for order in self.active_quotes[contract_id]:
                if (order.symbol == trade.symbol and 
                    order.side == trade.side and 
                    order.price == trade.price):
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = trade.quantity
                    break
        
        self.logger.info(f"Executed trade: {trade.side.value} {trade.quantity} {trade.symbol} @ {trade.price}")
        
        # Trigger immediate hedging check
        self._check_hedging_requirement()
    
    def _check_hedging_requirement(self):
        """Check if hedging is required and execute if needed."""
        hedge_orders = self.delta_hedger.run_hedging_cycle()
        if hedge_orders:
            for order in hedge_orders:
                # In a real system, these would be sent to the exchange
                self.logger.info(f"Hedge order: {order.side.value} {order.quantity} {order.symbol} @ {order.price}")
    
    def run_risk_checks(self):
        """Run comprehensive risk checks."""
        current_time = datetime.now()
        
        # Only run risk checks every minute
        if (current_time - self.last_risk_check).total_seconds() < 60:
            return
        
        risk_metrics = self.delta_hedger.get_risk_metrics()
        
        # Check delta exposure
        delta_limit = self.config.initial_capital * self.config.max_delta_exposure
        if abs(risk_metrics['total_delta']) > delta_limit:
            self.logger.warning(f"Delta exposure exceeded: {risk_metrics['total_delta']:.2f}")
            self._emergency_hedge()
        
        # Check gamma exposure
        gamma_limit = self.config.initial_capital * self.config.max_gamma_exposure
        if abs(risk_metrics['total_gamma']) > gamma_limit:
            self.logger.warning(f"Gamma exposure exceeded: {risk_metrics['total_gamma']:.2f}")
        
        # Log risk metrics
        self.logger.info(f"Risk metrics: Delta={risk_metrics['total_delta']:.2f}, "
                        f"Gamma={risk_metrics['total_gamma']:.2f}, "
                        f"PV={risk_metrics['portfolio_value']:.0f}")
        
        self.last_risk_check = current_time
    
    def _emergency_hedge(self):
        """Execute emergency hedging to reduce risk."""
        self.logger.warning("Executing emergency hedge")
        hedge_orders = self.delta_hedger.run_hedging_cycle()
        if hedge_orders:
            for order in hedge_orders:
                self.logger.warning(f"Emergency hedge: {order.side.value} {order.quantity} {order.symbol}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        risk_metrics = self.delta_hedger.get_risk_metrics()
        portfolio_greeks = self.delta_hedger.get_portfolio_greeks()
        
        # Calculate performance metrics
        runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        total_trades = len(self.trades_executed)
        
        return {
            'runtime_hours': runtime_hours,
            'total_trades': total_trades,
            'portfolio_value': risk_metrics['portfolio_value'],
            'cash': risk_metrics['cash'],
            'pnl': risk_metrics['portfolio_value'] - self.config.initial_capital,
            'pnl_pct': ((risk_metrics['portfolio_value'] / self.config.initial_capital) - 1) * 100,
            'positions': len(self.delta_hedger.portfolio.positions),
            'active_contracts': len(self.option_contracts),
            'greeks': {
                'delta': portfolio_greeks.delta,
                'gamma': portfolio_greeks.gamma,
                'theta': portfolio_greeks.theta,
                'vega': portfolio_greeks.vega,
                'rho': portfolio_greeks.rho
            },
            'risk_utilization': {
                'delta_pct': abs(portfolio_greeks.delta) / (self.config.initial_capital * self.config.max_delta_exposure) * 100,
                'gamma_pct': abs(portfolio_greeks.gamma) / (self.config.initial_capital * self.config.max_gamma_exposure) * 100
            }
        }
    
    async def run_market_making_loop(self):
        """Main market making loop."""
        self.is_running = True
        self.logger.info("Starting market making loop")
        
        while self.is_running:
            try:
                # Update quotes
                self.update_quotes()
                
                # Run risk checks
                self.run_risk_checks()
                
                # Check for gamma scalping opportunities
                self._check_gamma_scalping()
                
                # Sleep before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in market making loop: {e}")
                await asyncio.sleep(5)
    
    def _check_gamma_scalping(self):
        """Check for gamma scalping opportunities."""
        for underlying in self.underlying_symbols:
            if underlying in self.market_data:
                # This would use historical prices in practice
                current_price = self.market_data[underlying].mid_price
                # Simplified - would maintain price history
                price_history = [current_price * 0.999, current_price]
                
                scalp_order = self.delta_hedger.gamma_scalp(underlying, price_history)
                if scalp_order:
                    self.logger.info(f"Gamma scalp opportunity: {scalp_order.side.value} {scalp_order.quantity} {scalp_order.symbol}")
    
    def stop(self):
        """Stop the market maker."""
        self.is_running = False
        self.logger.info("Market maker stopped")
        
        # Cancel all active quotes
        for contract_id in self.active_quotes:
            self._cancel_quotes(contract_id)
        
        # Print final summary
        summary = self.get_portfolio_summary()
        self.logger.info(f"Final Summary: PnL={summary['pnl']:.2f} ({summary['pnl_pct']:.2f}%), "
                        f"Trades={summary['total_trades']}, Runtime={summary['runtime_hours']:.1f}h")