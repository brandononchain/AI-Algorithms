"""
Main entry point for the Quant Delta Market Maker system.
"""
import asyncio
import logging
from datetime import datetime, date
from typing import Dict, List

from data_models import (
    MarketMakerConfig, OptionContract, OptionType, MarketData
)
from market_maker import QuantDeltaMarketMaker
from risk_manager import RiskManager
from monitoring import MarketMakerMonitor
from backtesting import run_sample_backtest


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('market_maker.log'),
            logging.StreamHandler()
        ]
    )


def create_sample_options() -> List[OptionContract]:
    """Create sample option contracts for testing."""
    contracts = []
    
    # SPY options with different strikes and expirations
    strikes = [380, 400, 420]
    expiry_date = date(2024, 12, 20)
    
    for strike in strikes:
        # Call option
        call_contract = OptionContract(
            symbol=f"SPY{strike}C",
            underlying="SPY",
            strike=strike,
            expiry=expiry_date,
            option_type=OptionType.CALL
        )
        contracts.append(call_contract)
        
        # Put option
        put_contract = OptionContract(
            symbol=f"SPY{strike}P",
            underlying="SPY",
            strike=strike,
            expiry=expiry_date,
            option_type=OptionType.PUT
        )
        contracts.append(put_contract)
    
    return contracts


def create_sample_market_data() -> Dict[str, MarketData]:
    """Create sample market data."""
    market_data = {}
    
    # SPY underlying
    market_data["SPY"] = MarketData(
        symbol="SPY",
        timestamp=datetime.now(),
        bid=399.95,
        ask=400.05,
        last=400.00,
        volume=1000000,
        implied_volatility=None
    )
    
    # Option market data (simplified)
    option_data = {
        "SPY380C": {"bid": 22.50, "ask": 23.00, "iv": 0.18},
        "SPY380P": {"bid": 2.45, "ask": 2.55, "iv": 0.19},
        "SPY400C": {"bid": 8.75, "ask": 9.25, "iv": 0.16},
        "SPY400P": {"bid": 8.70, "ask": 9.20, "iv": 0.16},
        "SPY420C": {"bid": 2.40, "ask": 2.60, "iv": 0.17},
        "SPY420P": {"bid": 22.35, "ask": 22.85, "iv": 0.18}
    }
    
    for symbol, data in option_data.items():
        market_data[symbol] = MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=data["bid"],
            ask=data["ask"],
            last=(data["bid"] + data["ask"]) / 2,
            volume=5000,
            implied_volatility=data["iv"]
        )
    
    return market_data


async def run_live_trading():
    """Run live trading simulation."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Quant Delta Market Maker")
    
    # Configuration
    config = MarketMakerConfig(
        initial_capital=1_000_000,
        max_delta_exposure=0.05,
        max_gamma_exposure=0.02,
        bid_ask_spread=0.015,
        max_position_size=500,
        risk_free_rate=0.05,
        rebalance_threshold=0.03,
        quote_refresh_interval=10
    )
    
    # Initialize components
    market_maker = QuantDeltaMarketMaker(config)
    risk_manager = RiskManager(config)
    
    # Add option contracts
    contracts = create_sample_options()
    for contract in contracts:
        market_maker.add_option_contract(contract)
    
    # Initialize with sample market data
    sample_data = create_sample_market_data()
    for symbol, data in sample_data.items():
        market_maker.update_market_data(symbol, data)
    
    logger.info(f"Initialized with {len(contracts)} option contracts")
    
    # Start monitoring (optional)
    monitor = MarketMakerMonitor(market_maker, risk_manager)
    monitor.start_monitoring()
    
    # Run market making
    try:
        await market_maker.run_market_making_loop()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        market_maker.stop()
        monitor.stop_monitoring()
        
        # Print final summary
        summary = market_maker.get_portfolio_summary()
        logger.info("="*50)
        logger.info("FINAL SUMMARY")
        logger.info("="*50)
        logger.info(f"Runtime: {summary['runtime_hours']:.1f} hours")
        logger.info(f"Total Trades: {summary['total_trades']}")
        logger.info(f"Final P&L: ${summary['pnl']:+,.2f} ({summary['pnl_pct']:+.2f}%)")
        logger.info(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
        logger.info(f"Active Positions: {summary['positions']}")
        logger.info(f"Final Delta: {summary['greeks']['delta']:+.2f}")
        logger.info(f"Final Gamma: {summary['greeks']['gamma']:+.2f}")


def run_interactive_demo():
    """Run interactive demo with menu options."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("QUANT DELTA MARKET MAKER")
    print("="*60)
    print("\nSelect an option:")
    print("1. Run Backtest")
    print("2. Start Live Trading Simulation")
    print("3. Launch Monitoring Dashboard")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\nRunning backtest...")
                result = run_sample_backtest()
                print(f"\nBacktest completed. Total return: {result.total_return_pct:.2f}%")
                
            elif choice == "2":
                print("\nStarting live trading simulation...")
                print("Press Ctrl+C to stop")
                asyncio.run(run_live_trading())
                
            elif choice == "3":
                print("\nLaunching monitoring dashboard...")
                print("Dashboard will open at http://localhost:8050")
                from monitoring import create_sample_monitor
                monitor = create_sample_monitor()
                monitor.start_monitoring()
                monitor.run_dashboard()
                
            elif choice == "4":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\nOperation interrupted.")
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "backtest":
            setup_logging()
            run_sample_backtest()
            
        elif command == "live":
            setup_logging()
            asyncio.run(run_live_trading())
            
        elif command == "dashboard":
            setup_logging()
            from monitoring import create_sample_monitor
            monitor = create_sample_monitor()
            monitor.start_monitoring()
            monitor.run_dashboard()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: backtest, live, dashboard")
    else:
        # Interactive mode
        run_interactive_demo()


if __name__ == "__main__":
    main()