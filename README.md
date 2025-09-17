# Quant Delta Market Maker Algorithm

A sophisticated algorithmic trading system for options market making with delta hedging capabilities.

## Features

- **Options Pricing**: Black-Scholes model with Greeks calculation
- **Delta Hedging**: Automatic delta-neutral portfolio management
- **Market Making**: Intelligent bid/ask quote generation
- **Risk Management**: Position limits and exposure controls
- **Real-time Monitoring**: Live P&L and Greeks tracking
- **Backtesting**: Historical simulation framework

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from market_maker import DeltaMarketMaker
from data_models import OptionContract

# Initialize market maker
mm = DeltaMarketMaker(
    capital=1000000,
    max_delta_exposure=0.1,
    bid_ask_spread=0.02
)

# Run market making strategy
mm.run()
```

## Architecture

- `data_models.py` - Core data structures
- `greeks_calculator.py` - Options pricing and Greeks
- `delta_hedging.py` - Delta hedging logic
- `market_maker.py` - Main market making strategy
- `risk_manager.py` - Risk controls and limits
- `backtesting.py` - Historical simulation
- `monitoring.py` - Real-time dashboard

## Risk Disclaimer

This is for educational purposes only. Trading involves substantial risk and may not be suitable for all investors.