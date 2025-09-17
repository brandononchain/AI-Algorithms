"""
Data models for the Quant Delta Market Maker system.
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class OptionContract:
    """Represents an options contract."""
    symbol: str
    underlying: str
    strike: float
    expiry: date
    option_type: OptionType
    multiplier: int = 100
    
    def __post_init__(self):
        self.contract_id = f"{self.underlying}_{self.strike}_{self.option_type.value}_{self.expiry}"
    
    @property
    def time_to_expiry(self) -> float:
        """Returns time to expiry in years."""
        today = date.today()
        days_to_expiry = (self.expiry - today).days
        return max(days_to_expiry / 365.25, 0.001)  # Minimum 1 day
    
    def __hash__(self):
        return hash(self.contract_id)
    
    def __eq__(self, other):
        return isinstance(other, OptionContract) and self.contract_id == other.contract_id


@dataclass
class MarketData:
    """Market data snapshot for an asset."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    implied_volatility: Optional[float] = None
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid_price if self.mid_price > 0 else 0


@dataclass
class Greeks:
    """Options Greeks."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    def __mul__(self, scalar: float):
        """Scale Greeks by a scalar (e.g., position size)."""
        return Greeks(
            delta=self.delta * scalar,
            gamma=self.gamma * scalar,
            theta=self.theta * scalar,
            vega=self.vega * scalar,
            rho=self.rho * scalar
        )
    
    def __add__(self, other):
        """Add Greeks together."""
        if not isinstance(other, Greeks):
            return NotImplemented
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho
        )


@dataclass
class Position:
    """Represents a position in an asset."""
    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_price: float
    timestamp: datetime
    contract: Optional[OptionContract] = None
    
    @property
    def market_value(self) -> float:
        """Calculate market value given current price."""
        # This will be updated with current market price
        return self.quantity * self.avg_price
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    def update_position(self, quantity_change: int, price: float):
        """Update position with new trade."""
        if self.quantity == 0:
            self.avg_price = price
        else:
            # Update average price
            total_cost = self.quantity * self.avg_price + quantity_change * price
            self.quantity += quantity_change
            if self.quantity != 0:
                self.avg_price = total_cost / self.quantity
            else:
                self.avg_price = 0
        
        self.timestamp = datetime.now()


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    contract: Optional[OptionContract] = None
    
    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL
    
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        return self.filled_quantity == self.quantity


@dataclass
class Portfolio:
    """Portfolio containing positions and cash."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: Dict[str, Order] = field(default_factory=dict)
    
    def add_position(self, position: Position):
        """Add or update a position."""
        if position.symbol in self.positions:
            existing = self.positions[position.symbol]
            existing.update_position(position.quantity, position.avg_price)
        else:
            self.positions[position.symbol] = position
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_portfolio_greeks(self, greeks_by_symbol: Dict[str, Greeks]) -> Greeks:
        """Calculate total portfolio Greeks."""
        total_greeks = Greeks(0, 0, 0, 0, 0)
        
        for symbol, position in self.positions.items():
            if symbol in greeks_by_symbol and position.contract:
                position_greeks = greeks_by_symbol[symbol] * position.quantity
                total_greeks = total_greeks + position_greeks
        
        return total_greeks
    
    def get_net_delta(self, greeks_by_symbol: Dict[str, Greeks]) -> float:
        """Get net portfolio delta."""
        return self.get_portfolio_greeks(greeks_by_symbol).delta


@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    contract: Optional[OptionContract] = None
    
    @property
    def notional_value(self) -> float:
        """Total value of the trade."""
        return self.quantity * self.price
    
    @property
    def net_value(self) -> float:
        """Net value after commission."""
        return self.notional_value - self.commission


@dataclass
class RiskMetrics:
    """Risk metrics for the portfolio."""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    var_95: float  # Value at Risk at 95% confidence
    max_drawdown: float
    sharpe_ratio: float
    
    def is_within_limits(self, max_delta: float, max_gamma: float) -> bool:
        """Check if risk metrics are within acceptable limits."""
        return (abs(self.total_delta) <= max_delta and 
                abs(self.total_gamma) <= max_gamma)


@dataclass
class MarketMakerConfig:
    """Configuration for the market maker."""
    initial_capital: float = 1_000_000
    max_delta_exposure: float = 0.1  # Maximum delta as % of capital
    max_gamma_exposure: float = 0.05  # Maximum gamma exposure
    bid_ask_spread: float = 0.02  # Target bid-ask spread
    max_position_size: int = 1000  # Maximum position size per contract
    risk_free_rate: float = 0.05  # Risk-free rate for pricing
    rebalance_threshold: float = 0.05  # Delta threshold for rebalancing
    quote_refresh_interval: int = 5  # Seconds between quote updates
    max_orders_per_symbol: int = 10  # Maximum pending orders per symbol


@dataclass
class PnLSnapshot:
    """P&L snapshot at a point in time."""
    timestamp: datetime
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    portfolio_value: float
    cash: float
    
    @property
    def return_pct(self) -> float:
        """Return as percentage of initial capital."""
        return self.total_pnl / self.portfolio_value if self.portfolio_value > 0 else 0