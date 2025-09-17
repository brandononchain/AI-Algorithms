"""
Example usage scenarios for the Quant Delta Market Maker.
"""
import asyncio
from datetime import date, datetime
from data_models import (
    MarketMakerConfig, OptionContract, OptionType, MarketData, Trade, OrderSide
)
from market_maker import QuantDeltaMarketMaker
from backtesting import run_sample_backtest
import logging


def example_1_basic_market_making():
    """Example 1: Basic market making setup."""
    print("\n" + "="*50)
    print("EXAMPLE 1: Basic Market Making Setup")
    print("="*50)
    
    # Create configuration
    config = MarketMakerConfig(
        initial_capital=500_000,
        max_delta_exposure=0.03,  # 3% max delta exposure
        bid_ask_spread=0.02,      # 2% bid-ask spread
        max_position_size=100,    # Max 100 contracts per position
        risk_free_rate=0.05
    )
    
    # Initialize market maker
    mm = QuantDeltaMarketMaker(config)
    
    # Create an option contract
    spy_call = OptionContract(
        symbol="SPY400C",
        underlying="SPY",
        strike=400,
        expiry=date(2024, 12, 20),
        option_type=OptionType.CALL
    )
    
    mm.add_option_contract(spy_call)
    
    # Add market data
    spy_data = MarketData(
        symbol="SPY",
        timestamp=datetime.now(),
        bid=399.90,
        ask=400.10,
        last=400.00,
        volume=100000
    )
    
    option_data = MarketData(
        symbol="SPY400C",
        timestamp=datetime.now(),
        bid=8.50,
        ask=9.00,
        last=8.75,
        volume=1000,
        implied_volatility=0.20
    )
    
    mm.update_market_data("SPY", spy_data)
    mm.update_market_data("SPY400C", option_data)
    
    # Generate quotes
    quotes = mm.generate_quotes(spy_call)
    if quotes:
        bid_order, ask_order = quotes
        print(f"Generated quotes for {spy_call.contract_id}:")
        print(f"  Bid: {bid_order.quantity} @ ${bid_order.price}")
        print(f"  Ask: {ask_order.quantity} @ ${ask_order.price}")
        
        # Calculate fair value
        fair_value = mm.calculate_fair_value(spy_call)
        print(f"  Fair Value: ${fair_value:.2f}")
        
        # Show Greeks
        greeks = mm.greeks_calc.calculate_greeks(
            400.0, 400, spy_call.time_to_expiry, 0.20, OptionType.CALL
        )
        print(f"  Delta: {greeks.delta:.3f}")
        print(f"  Gamma: {greeks.gamma:.3f}")
        print(f"  Theta: {greeks.theta:.3f}")
        print(f"  Vega: {greeks.vega:.3f}")
    
    print("✓ Basic market making setup completed")


def example_2_delta_hedging():
    """Example 2: Delta hedging demonstration."""
    print("\n" + "="*50)
    print("EXAMPLE 2: Delta Hedging Demonstration")
    print("="*50)
    
    config = MarketMakerConfig(initial_capital=1_000_000)
    mm = QuantDeltaMarketMaker(config)
    
    # Create multiple option contracts
    contracts = [
        OptionContract("SPY390C", "SPY", 390, date(2024, 12, 20), OptionType.CALL),
        OptionContract("SPY400C", "SPY", 400, date(2024, 12, 20), OptionType.CALL),
        OptionContract("SPY410C", "SPY", 410, date(2024, 12, 20), OptionType.CALL),
    ]
    
    for contract in contracts:
        mm.add_option_contract(contract)
    
    # Simulate some trades to create positions
    trades = [
        Trade("t1", "SPY390C", OrderSide.SELL, 50, 15.00, datetime.now(), 1.0, contracts[0]),
        Trade("t2", "SPY400C", OrderSide.BUY, 30, 8.50, datetime.now(), 1.0, contracts[1]),
        Trade("t3", "SPY410C", OrderSide.SELL, 20, 4.00, datetime.now(), 1.0, contracts[2]),
    ]
    
    # Update market data
    mm.update_market_data("SPY", MarketData("SPY", datetime.now(), 399.50, 400.50, 400.0, 100000))
    
    print("Processing trades and calculating delta exposure...")
    for trade in trades:
        mm.process_trade_execution(trade)
        print(f"  {trade.side.value.upper()} {trade.quantity} {trade.symbol} @ ${trade.price}")
    
    # Calculate portfolio delta
    portfolio_delta = mm.delta_hedger.calculate_portfolio_delta()
    print(f"\nPortfolio Delta: {portfolio_delta:.2f}")
    
    # Calculate hedge requirements
    hedge_req = mm.delta_hedger.calculate_hedge_requirement()
    print(f"Hedge Requirements: {hedge_req}")
    
    if hedge_req:
        print("Delta hedging would be triggered!")
    
    print("✓ Delta hedging demonstration completed")


def example_3_risk_management():
    """Example 3: Risk management features."""
    print("\n" + "="*50)
    print("EXAMPLE 3: Risk Management Features")
    print("="*50)
    
    # Strict risk limits for demonstration
    config = MarketMakerConfig(
        initial_capital=100_000,  # Smaller capital
        max_delta_exposure=0.02,  # Very low delta limit
        max_gamma_exposure=0.01,  # Very low gamma limit
        max_position_size=50      # Small position size
    )
    
    mm = QuantDeltaMarketMaker(config)
    
    # Create a high-delta option
    itm_call = OptionContract(
        "SPY380C", "SPY", 380, date(2024, 12, 20), OptionType.CALL
    )
    mm.add_option_contract(itm_call)
    
    # Update market data
    mm.update_market_data("SPY", MarketData("SPY", datetime.now(), 399.50, 400.50, 400.0, 100000))
    mm.update_market_data("SPY380C", MarketData("SPY380C", datetime.now(), 21.50, 22.00, 21.75, 1000, 0.18))
    
    # Try to create a large position that would breach risk limits
    large_trade = Trade(
        "risk_test", "SPY380C", OrderSide.SELL, 100, 21.75, datetime.now(), 1.0, itm_call
    )
    
    print(f"Attempting large trade: SELL 100 SPY380C @ $21.75")
    
    # This would normally trigger risk checks
    portfolio_summary = mm.get_portfolio_summary()
    print(f"Current Portfolio Delta: {portfolio_summary['greeks']['delta']:.2f}")
    print(f"Delta Limit: {config.initial_capital * config.max_delta_exposure:.0f}")
    
    # Calculate what the delta would be after this trade
    greeks = mm.greeks_calc.calculate_greeks(400, 380, itm_call.time_to_expiry, 0.18, OptionType.CALL)
    trade_delta_impact = greeks.delta * -100  # Negative because we're selling
    
    print(f"Trade Delta Impact: {trade_delta_impact:.2f}")
    print(f"Would breach limits: {'YES' if abs(trade_delta_impact) > config.initial_capital * config.max_delta_exposure else 'NO'}")
    
    print("✓ Risk management demonstration completed")


def example_4_backtesting():
    """Example 4: Run a backtest."""
    print("\n" + "="*50)
    print("EXAMPLE 4: Backtesting")
    print("="*50)
    
    print("Running sample backtest (this may take a moment)...")
    
    # Run the sample backtest
    result = run_sample_backtest()
    
    # Display key results
    print(f"\nBacktest Results Summary:")
    print(f"  Period: {result.start_date} to {result.end_date}")
    print(f"  Total Return: {result.total_return_pct:.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: ${result.max_drawdown:,.2f}")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.1%}")
    print(f"  Final Greeks - Delta: {result.greeks_summary['delta']:.2f}")
    
    print("✓ Backtesting demonstration completed")


async def example_5_live_simulation():
    """Example 5: Short live simulation."""
    print("\n" + "="*50)
    print("EXAMPLE 5: Live Simulation (10 seconds)")
    print("="*50)
    
    config = MarketMakerConfig(
        initial_capital=250_000,
        quote_refresh_interval=2  # Update quotes every 2 seconds
    )
    
    mm = QuantDeltaMarketMaker(config)
    
    # Add a simple option
    contract = OptionContract("SPY400C", "SPY", 400, date(2024, 12, 20), OptionType.CALL)
    mm.add_option_contract(contract)
    
    # Add market data
    mm.update_market_data("SPY", MarketData("SPY", datetime.now(), 399.90, 400.10, 400.0, 100000))
    mm.update_market_data("SPY400C", MarketData("SPY400C", datetime.now(), 8.40, 8.60, 8.50, 1000, 0.20))
    
    print("Starting 10-second live simulation...")
    print("Market maker will generate quotes every 2 seconds")
    
    # Run for 10 seconds
    start_time = datetime.now()
    iteration = 0
    
    while (datetime.now() - start_time).total_seconds() < 10:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # Update quotes
        mm.update_quotes()
        
        # Show active quotes
        for contract_id, quotes in mm.active_quotes.items():
            if quotes:
                for quote in quotes:
                    print(f"  {quote.side.value.upper()}: {quote.quantity} @ ${quote.price}")
        
        # Show portfolio status
        summary = mm.get_portfolio_summary()
        print(f"  Portfolio Value: ${summary['portfolio_value']:,.2f}")
        print(f"  P&L: ${summary['pnl']:+,.2f}")
        
        await asyncio.sleep(2)
    
    mm.stop()
    print("\n✓ Live simulation completed")


def run_all_examples():
    """Run all examples."""
    # Setup logging to reduce noise
    logging.getLogger().setLevel(logging.WARNING)
    
    print("QUANT DELTA MARKET MAKER - EXAMPLES")
    print("This will demonstrate key features of the system")
    
    try:
        example_1_basic_market_making()
        example_2_delta_hedging()
        example_3_risk_management()
        
        # Ask user if they want to run longer examples
        print(f"\nThe remaining examples involve backtesting and live simulation.")
        print("These may take longer to complete.")
        
        choice = input("Run backtesting example? (y/n): ").lower().strip()
        if choice == 'y':
            example_4_backtesting()
        
        choice = input("Run live simulation example? (y/n): ").lower().strip()
        if choice == 'y':
            asyncio.run(example_5_live_simulation())
        
        print(f"\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe Quant Delta Market Maker system includes:")
        print("✓ Options pricing with Black-Scholes model")
        print("✓ Greeks calculation (Delta, Gamma, Theta, Vega, Rho)")
        print("✓ Automatic delta hedging")
        print("✓ Market making with intelligent bid/ask spreads")
        print("✓ Comprehensive risk management")
        print("✓ Backtesting framework")
        print("✓ Real-time monitoring dashboard")
        print("✓ Position and portfolio management")
        
        print(f"\nTo get started:")
        print("1. Run 'python main.py' for interactive mode")
        print("2. Run 'python main.py backtest' for backtesting")
        print("3. Run 'python main.py live' for live trading simulation")
        print("4. Run 'python main.py dashboard' for monitoring dashboard")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")


if __name__ == "__main__":
    run_all_examples()