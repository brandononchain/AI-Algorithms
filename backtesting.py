"""
Backtesting framework for the Quant Delta Market Maker.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

from data_models import (
    OptionContract, OptionType, MarketData, Trade, OrderSide,
    MarketMakerConfig, Portfolio, Position
)
from market_maker import QuantDeltaMarketMaker
from greeks_calculator import GreeksCalculator
from risk_manager import RiskManager


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    win_rate: float
    avg_trade_pnl: float
    greeks_summary: Dict[str, float]
    daily_pnl: List[float]
    daily_returns: List[float]
    risk_events: int


class MarketDataSimulator:
    """Simulates market data for backtesting."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
    
    def generate_underlying_path(
        self,
        initial_price: float,
        volatility: float,
        drift: float,
        days: int,
        dt: float = 1/252  # Daily steps
    ) -> List[float]:
        """Generate geometric Brownian motion path for underlying."""
        
        prices = [initial_price]
        
        for _ in range(days):
            prev_price = prices[-1]
            
            # Geometric Brownian Motion
            random_shock = np.random.normal(0, 1)
            price_change = prev_price * (drift * dt + volatility * np.sqrt(dt) * random_shock)
            new_price = prev_price + price_change
            
            # Ensure price stays positive
            new_price = max(new_price, 0.01)
            prices.append(new_price)
        
        return prices[1:]  # Remove initial price
    
    def generate_option_prices(
        self,
        contract: OptionContract,
        underlying_prices: List[float],
        greeks_calc: GreeksCalculator,
        base_volatility: float = 0.2
    ) -> List[MarketData]:
        """Generate option market data based on underlying prices."""
        
        option_data = []
        
        for i, underlying_price in enumerate(underlying_prices):
            # Calculate days to expiry
            current_date = date.today() + timedelta(days=i)
            days_to_expiry = (contract.expiry - current_date).days
            time_to_expiry = max(days_to_expiry / 365.25, 0.001)
            
            # Add some volatility smile/skew
            moneyness = underlying_price / contract.strike
            vol_adjustment = 1 + 0.1 * (1 - moneyness) if contract.option_type == OptionType.PUT else 1
            implied_vol = base_volatility * vol_adjustment
            
            # Calculate theoretical price
            theoretical_price = greeks_calc.black_scholes_price(
                underlying_price,
                contract.strike,
                time_to_expiry,
                implied_vol,
                contract.option_type
            )
            
            # Add bid-ask spread (wider for illiquid options)
            liquidity_factor = min(time_to_expiry * 4, 1.0)  # Less liquid near expiry
            spread_pct = 0.02 + (1 - liquidity_factor) * 0.03
            half_spread = theoretical_price * spread_pct / 2
            
            bid = max(theoretical_price - half_spread, 0.01)
            ask = theoretical_price + half_spread
            
            # Add some randomness to market prices
            noise = np.random.normal(0, theoretical_price * 0.01)
            market_price = max(theoretical_price + noise, 0.01)
            
            market_data = MarketData(
                symbol=contract.contract_id,
                timestamp=datetime.combine(current_date, datetime.min.time()),
                bid=round(bid, 2),
                ask=round(ask, 2),
                last=round(market_price, 2),
                volume=np.random.randint(100, 1000),
                implied_volatility=implied_vol
            )
            
            option_data.append(market_data)
        
        return option_data


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self.greeks_calc = GreeksCalculator(config.risk_free_rate)
        self.market_sim = MarketDataSimulator()
        self.logger = logging.getLogger(__name__)
        
        # Results tracking
        self.trades_log: List[Trade] = []
        self.pnl_history: List[Dict] = []
        self.risk_events: List[Dict] = []
        
    def run_backtest(
        self,
        start_date: date,
        end_date: date,
        underlying_config: Dict[str, Any],
        option_configs: List[Dict[str, Any]]
    ) -> BacktestResult:
        """
        Run a complete backtest.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            underlying_config: Configuration for underlying simulation
            option_configs: List of option contract configurations
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Calculate simulation parameters
        total_days = (end_date - start_date).days
        
        # Generate underlying price path
        underlying_prices = self.market_sim.generate_underlying_path(
            initial_price=underlying_config['initial_price'],
            volatility=underlying_config['volatility'],
            drift=underlying_config['drift'],
            days=total_days
        )
        
        # Create option contracts
        option_contracts = []
        for opt_config in option_configs:
            contract = OptionContract(
                symbol=opt_config['symbol'],
                underlying=underlying_config['symbol'],
                strike=opt_config['strike'],
                expiry=date.fromisoformat(opt_config['expiry']),
                option_type=OptionType(opt_config['option_type'])
            )
            option_contracts.append(contract)
        
        # Generate option market data
        all_option_data = {}
        for contract in option_contracts:
            option_data = self.market_sim.generate_option_prices(
                contract, underlying_prices, self.greeks_calc
            )
            all_option_data[contract.contract_id] = option_data
        
        # Initialize market maker
        market_maker = QuantDeltaMarketMaker(self.config)
        risk_manager = RiskManager(self.config)
        
        # Add contracts to market maker
        for contract in option_contracts:
            market_maker.add_option_contract(contract)
        
        # Run simulation day by day
        daily_results = []
        
        for day in range(total_days):
            current_date = start_date + timedelta(days=day)
            underlying_price = underlying_prices[day]
            
            # Update underlying market data
            underlying_data = MarketData(
                symbol=underlying_config['symbol'],
                timestamp=datetime.combine(current_date, datetime.min.time()),
                bid=underlying_price * 0.9995,
                ask=underlying_price * 1.0005,
                last=underlying_price,
                volume=np.random.randint(10000, 100000)
            )
            market_maker.update_market_data(underlying_config['symbol'], underlying_data)
            
            # Update option market data
            for contract in option_contracts:
                if day < len(all_option_data[contract.contract_id]):
                    option_data = all_option_data[contract.contract_id][day]
                    market_maker.update_market_data(contract.contract_id, option_data)
            
            # Generate quotes and simulate trading
            trades_today = self._simulate_trading_day(market_maker, risk_manager)
            
            # Calculate daily P&L and metrics
            daily_result = self._calculate_daily_metrics(
                market_maker, risk_manager, current_date, trades_today
            )
            daily_results.append(daily_result)
            
            # Log progress
            if day % 30 == 0:  # Log monthly
                self.logger.info(f"Day {day}/{total_days}: PnL={daily_result['pnl']:.2f}, "
                               f"Portfolio Value={daily_result['portfolio_value']:.2f}")
        
        # Compile final results
        result = self._compile_backtest_results(
            start_date, end_date, daily_results, market_maker
        )
        
        self.logger.info(f"Backtest completed. Total PnL: {result.total_pnl:.2f} "
                        f"({result.total_return_pct:.2f}%)")
        
        return result
    
    def _simulate_trading_day(
        self,
        market_maker: QuantDeltaMarketMaker,
        risk_manager: RiskManager
    ) -> List[Trade]:
        """Simulate trading activity for one day."""
        
        trades_today = []
        
        # Update quotes
        market_maker.update_quotes()
        
        # Simulate some trading activity
        for contract_id, quotes in market_maker.active_quotes.items():
            if not quotes:
                continue
            
            contract = market_maker.option_contracts.get(contract_id)
            if not contract:
                continue
            
            # Simulate market participants hitting our quotes
            hit_probability = 0.1  # 10% chance per quote per day
            
            for quote in quotes:
                if np.random.random() < hit_probability:
                    # Someone hit our quote
                    trade = Trade(
                        trade_id=f"trade_{len(self.trades_log)}",
                        symbol=quote.symbol,
                        side=OrderSide.SELL if quote.side == OrderSide.BUY else OrderSide.BUY,
                        quantity=min(quote.quantity, np.random.randint(1, 20)),
                        price=quote.price,
                        timestamp=datetime.now(),
                        commission=1.0,
                        contract=contract
                    )
                    
                    # Check if trade passes risk management
                    trade_proposal = {
                        'symbol': trade.symbol,
                        'quantity': trade.quantity,
                        'side': trade.side.value,
                        'price': trade.price
                    }
                    
                    approved, reason = risk_manager.assess_trade_risk(
                        trade_proposal,
                        market_maker.delta_hedger.portfolio,
                        market_maker.market_data,
                        market_maker.delta_hedger.greeks_cache
                    )
                    
                    if approved:
                        market_maker.process_trade_execution(trade)
                        trades_today.append(trade)
                        self.trades_log.append(trade)
                    else:
                        self.logger.warning(f"Trade rejected: {reason}")
        
        # Run risk checks
        risk_limits = risk_manager.check_risk_limits(
            market_maker.delta_hedger.portfolio,
            market_maker.market_data,
            market_maker.delta_hedger.greeks_cache
        )
        
        # Count risk events
        risk_violations = [l for l in risk_limits if l.is_breached]
        if risk_violations:
            self.risk_events.extend([{
                'timestamp': datetime.now(),
                'type': 'limit_breach',
                'description': f"{len(risk_violations)} limits breached"
            }])
        
        return trades_today
    
    def _calculate_daily_metrics(
        self,
        market_maker: QuantDeltaMarketMaker,
        risk_manager: RiskManager,
        date: date,
        trades: List[Trade]
    ) -> Dict[str, Any]:
        """Calculate metrics for a single day."""
        
        portfolio_summary = market_maker.get_portfolio_summary()
        
        daily_result = {
            'date': date,
            'portfolio_value': portfolio_summary['portfolio_value'],
            'pnl': portfolio_summary['pnl'],
            'pnl_pct': portfolio_summary['pnl_pct'],
            'trades_count': len(trades),
            'delta': portfolio_summary['greeks']['delta'],
            'gamma': portfolio_summary['greeks']['gamma'],
            'theta': portfolio_summary['greeks']['theta'],
            'vega': portfolio_summary['greeks']['vega'],
            'positions_count': portfolio_summary['positions']
        }
        
        self.pnl_history.append(daily_result)
        return daily_result
    
    def _compile_backtest_results(
        self,
        start_date: date,
        end_date: date,
        daily_results: List[Dict],
        market_maker: QuantDeltaMarketMaker
    ) -> BacktestResult:
        """Compile final backtest results."""
        
        if not daily_results:
            raise ValueError("No daily results to compile")
        
        # Extract metrics
        pnl_values = [d['pnl'] for d in daily_results]
        portfolio_values = [d['portfolio_value'] for d in daily_results]
        
        final_pnl = pnl_values[-1]
        total_return_pct = (portfolio_values[-1] / self.config.initial_capital - 1) * 100
        
        # Calculate max drawdown
        peak_value = self.config.initial_capital
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak_value:
                peak_value = value
            drawdown = peak_value - value
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        if daily_returns and np.std(daily_returns) > 0:
            excess_return = np.mean(daily_returns) - (self.config.risk_free_rate / 365)
            sharpe_ratio = excess_return / np.std(daily_returns) * np.sqrt(365)
        else:
            sharpe_ratio = 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades_log if self._calculate_trade_pnl(t) > 0]
        win_rate = len(winning_trades) / len(self.trades_log) if self.trades_log else 0
        avg_trade_pnl = np.mean([self._calculate_trade_pnl(t) for t in self.trades_log]) if self.trades_log else 0
        
        # Greeks summary
        final_summary = market_maker.get_portfolio_summary()
        greeks_summary = final_summary['greeks']
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.initial_capital,
            final_capital=portfolio_values[-1],
            total_pnl=final_pnl,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=len(self.trades_log),
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            greeks_summary=greeks_summary,
            daily_pnl=pnl_values,
            daily_returns=daily_returns,
            risk_events=len(self.risk_events)
        )
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a single trade (simplified)."""
        # This is a simplified calculation
        # In practice, you'd need to track the full lifecycle of each position
        multiplier = trade.contract.multiplier if trade.contract else 1
        base_pnl = trade.quantity * trade.price * multiplier
        
        if trade.side == OrderSide.SELL:
            return base_pnl - trade.commission
        else:
            return -base_pnl - trade.commission
    
    def plot_backtest_results(self, result: BacktestResult, save_path: Optional[str] = None):
        """Plot backtest results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Quant Delta Market Maker Backtest Results', fontsize=16)
        
        # Portfolio value over time
        days = range(len(result.daily_pnl))
        portfolio_values = [result.initial_capital + pnl for pnl in result.daily_pnl]
        
        axes[0, 0].plot(days, portfolio_values, linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Daily returns distribution
        axes[0, 1].hist(result.daily_returns, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Daily Returns Distribution')
        axes[0, 1].set_xlabel('Daily Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum(result.daily_pnl)
        axes[1, 0].plot(days, cumulative_pnl, linewidth=2, color='green')
        axes[1, 0].set_title('Cumulative P&L')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel('Cumulative P&L ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics text
        metrics_text = f"""
        Total Return: {result.total_return_pct:.2f}%
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        Max Drawdown: ${result.max_drawdown:,.2f}
        Total Trades: {result.total_trades}
        Win Rate: {result.win_rate:.1%}
        Avg Trade P&L: ${result.avg_trade_pnl:.2f}
        Risk Events: {result.risk_events}
        """
        
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def run_sample_backtest():
    """Run a sample backtest with example parameters."""
    
    # Configuration
    config = MarketMakerConfig(
        initial_capital=1_000_000,
        max_delta_exposure=0.05,
        max_gamma_exposure=0.02,
        bid_ask_spread=0.02,
        max_position_size=500,
        risk_free_rate=0.05
    )
    
    # Underlying configuration
    underlying_config = {
        'symbol': 'SPY',
        'initial_price': 400.0,
        'volatility': 0.15,
        'drift': 0.08
    }
    
    # Option configurations
    option_configs = [
        {
            'symbol': 'SPY_400_CALL',
            'strike': 400,
            'expiry': '2024-12-20',
            'option_type': 'call'
        },
        {
            'symbol': 'SPY_400_PUT',
            'strike': 400,
            'expiry': '2024-12-20',
            'option_type': 'put'
        },
        {
            'symbol': 'SPY_420_CALL',
            'strike': 420,
            'expiry': '2024-12-20',
            'option_type': 'call'
        },
        {
            'symbol': 'SPY_380_PUT',
            'strike': 380,
            'expiry': '2024-12-20',
            'option_type': 'put'
        }
    ]
    
    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run_backtest(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        underlying_config=underlying_config,
        option_configs=option_configs
    )
    
    # Display results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Capital: ${result.final_capital:,.2f}")
    print(f"Total P&L: ${result.total_pnl:,.2f}")
    print(f"Total Return: {result.total_return_pct:.2f}%")
    print(f"Max Drawdown: ${result.max_drawdown:,.2f}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Avg Trade P&L: ${result.avg_trade_pnl:.2f}")
    print(f"Risk Events: {result.risk_events}")
    
    # Plot results
    engine.plot_backtest_results(result)
    
    return result


if __name__ == "__main__":
    run_sample_backtest()