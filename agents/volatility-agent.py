"""
Volatility Trading Agent

Strategies that trade on volatility patterns:
- Volatility breakouts
- Volatility mean reversion  
- VIX-based strategies
- Volatility surface arbitrage
"""

import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class VolatilityBreakoutAgent(BaseAgent):
    """
    Volatility breakout strategy that trades when volatility breaks
    above/below historical ranges.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.vol_lookback = self.config.get("vol_lookback", 20)
        self.breakout_threshold = self.config.get("breakout_threshold", 2.0)  # Standard deviations
        self.min_vol_change = self.config.get("min_vol_change", 0.5)  # Minimum volatility change
        self.holding_period = self.config.get("holding_period", 5)  # Days to hold position
        self.vol_estimation_method = self.config.get("vol_estimation_method", "close_to_close")
        
    def calculate_volatility(self, market_data: pd.DataFrame, method: str = "close_to_close") -> pd.Series:
        """Calculate volatility using different methods"""
        if method == "close_to_close":
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
            
        elif method == "parkinson" and all(col in market_data.columns for col in ['high', 'low']):
            # Parkinson volatility estimator
            high = market_data['high']
            low = market_data['low']
            hl_ratio = np.log(high / low)
            parkinson_var = (hl_ratio ** 2) / (4 * np.log(2))
            volatility = np.sqrt(parkinson_var.rolling(self.vol_lookback).mean() * 252)
            
        elif method == "garman_klass" and all(col in market_data.columns for col in ['high', 'low', 'open', 'close']):
            # Garman-Klass volatility estimator
            high = market_data['high']
            low = market_data['low']
            open_price = market_data['open']
            close = market_data['close']
            
            gk_var = (0.5 * (np.log(high / low) ** 2) - 
                     (2 * np.log(2) - 1) * (np.log(close / open_price) ** 2))
            volatility = np.sqrt(gk_var.rolling(self.vol_lookback).mean() * 252)
            
        else:
            # Default to close-to-close
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
            
        return volatility
    
    def detect_volatility_breakout(self, volatility: pd.Series) -> pd.Series:
        """Detect volatility breakouts"""
        vol_mean = volatility.rolling(self.vol_lookback * 2).mean()
        vol_std = volatility.rolling(self.vol_lookback * 2).std()
        
        # Z-score of current volatility
        vol_zscore = (volatility - vol_mean) / vol_std
        
        # Breakout signals
        breakout_signals = pd.Series(0, index=volatility.index)
        breakout_signals[vol_zscore > self.breakout_threshold] = 1  # High vol breakout
        breakout_signals[vol_zscore < -self.breakout_threshold] = -1  # Low vol breakout
        
        return breakout_signals
    
    def generate_signal(self, market_data: pd.DataFrame) -> str:
        """Generate volatility breakout signal"""
        if len(market_data) < self.vol_lookback * 3:
            return 'HOLD'
        
        # Calculate volatility
        volatility = self.calculate_volatility(market_data, self.vol_estimation_method)
        
        # Detect breakouts
        breakout_signals = self.detect_volatility_breakout(volatility)
        
        current_signal = breakout_signals.iloc[-1]
        current_vol = volatility.iloc[-1]
        prev_vol = volatility.iloc[-2] if len(volatility) > 1 else current_vol
        
        # Check for minimum volatility change
        vol_change = abs(current_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
        
        if vol_change < self.min_vol_change:
            return 'HOLD'
        
        # Generate trading signal
        if current_signal == 1:
            return 'BUY'  # High volatility breakout - expect continuation
        elif current_signal == -1:
            return 'SELL'  # Low volatility breakout - expect mean reversion
        else:
            return 'HOLD'


class VolatilityMeanReversionAgent(BaseAgent):
    """
    Volatility mean reversion strategy that trades when volatility
    is expected to revert to its long-term mean.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.short_vol_window = self.config.get("short_vol_window", 10)
        self.long_vol_window = self.config.get("long_vol_window", 50)
        self.reversion_threshold = self.config.get("reversion_threshold", 1.5)
        self.vol_percentile_high = self.config.get("vol_percentile_high", 80)
        self.vol_percentile_low = self.config.get("vol_percentile_low", 20)
        
    def calculate_volatility_regime(self, market_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Identify volatility regime and mean reversion opportunities"""
        returns = market_data['close'].pct_change()
        
        # Short and long-term volatility
        short_vol = returns.rolling(self.short_vol_window).std() * np.sqrt(252)
        long_vol = returns.rolling(self.long_vol_window).std() * np.sqrt(252)
        
        # Volatility ratio
        vol_ratio = short_vol / long_vol
        
        # Historical percentiles
        vol_percentiles = short_vol.rolling(self.long_vol_window * 2).rank(pct=True) * 100
        
        return short_vol, long_vol, vol_ratio, vol_percentiles
    
    def generate_signal(self, market_data: pd.DataFrame) -> str:
        """Generate volatility mean reversion signal"""
        if len(market_data) < self.long_vol_window * 2:
            return 'HOLD'
        
        short_vol, long_vol, vol_ratio, vol_percentiles = self.calculate_volatility_regime(market_data)
        
        current_vol_ratio = vol_ratio.iloc[-1]
        current_percentile = vol_percentiles.iloc[-1]
        
        # Mean reversion signals
        if (current_vol_ratio > self.reversion_threshold and 
            current_percentile > self.vol_percentile_high):
            return 'SELL'  # High volatility, expect reversion down
        elif (current_vol_ratio < (1 / self.reversion_threshold) and 
              current_percentile < self.vol_percentile_low):
            return 'BUY'   # Low volatility, expect reversion up
        else:
            return 'HOLD'


class VIXBasedAgent(BaseAgent):
    """
    VIX-based trading strategy (simulated VIX from price data).
    Trades based on fear/greed cycles in the market.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.vix_window = self.config.get("vix_window", 20)
        self.vix_high_threshold = self.config.get("vix_high_threshold", 30)  # High fear
        self.vix_low_threshold = self.config.get("vix_low_threshold", 15)   # Low fear/complacency
        self.vix_spike_threshold = self.config.get("vix_spike_threshold", 1.5)  # VIX spike multiplier
        
    def calculate_synthetic_vix(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate synthetic VIX from price data"""
        returns = market_data['close'].pct_change()
        
        # Rolling volatility (annualized)
        rolling_vol = returns.rolling(self.vix_window).std() * np.sqrt(252) * 100
        
        # Apply VIX-like scaling (VIX tends to be higher than realized vol)
        synthetic_vix = rolling_vol * 1.2  # Scaling factor
        
        return synthetic_vix
    
    def detect_vix_spikes(self, vix: pd.Series) -> pd.Series:
        """Detect VIX spikes that often mark market bottoms"""
        vix_ma = vix.rolling(self.vix_window).mean()
        vix_spikes = vix > (vix_ma * self.vix_spike_threshold)
        
        return vix_spikes
    
    def generate_signal(self, market_data: pd.DataFrame) -> str:
        """Generate VIX-based trading signal"""
        if len(market_data) < self.vix_window * 2:
            return 'HOLD'
        
        synthetic_vix = self.calculate_synthetic_vix(market_data)
        vix_spikes = self.detect_vix_spikes(synthetic_vix)
        
        current_vix = synthetic_vix.iloc[-1]
        current_spike = vix_spikes.iloc[-1]
        
        # VIX-based signals
        if current_spike or current_vix > self.vix_high_threshold:
            return 'BUY'   # High fear - contrarian buy
        elif current_vix < self.vix_low_threshold:
            return 'SELL'  # Low fear/complacency - expect volatility increase
        else:
            return 'HOLD'


class VolatilitySurfaceAgent(BaseAgent):
    """
    Volatility surface arbitrage strategy that looks for
    inconsistencies in implied vs realized volatility.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.short_term_window = self.config.get("short_term_window", 5)
        self.medium_term_window = self.config.get("medium_term_window", 20)
        self.long_term_window = self.config.get("long_term_window", 60)
        self.vol_spread_threshold = self.config.get("vol_spread_threshold", 0.05)
        
    def calculate_term_structure(self, market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volatility term structure"""
        returns = market_data['close'].pct_change()
        
        vol_structure = {
            'short_term': returns.rolling(self.short_term_window).std() * np.sqrt(252),
            'medium_term': returns.rolling(self.medium_term_window).std() * np.sqrt(252),
            'long_term': returns.rolling(self.long_term_window).std() * np.sqrt(252)
        }
        
        return vol_structure
    
    def detect_term_structure_anomalies(self, vol_structure: Dict[str, pd.Series]) -> pd.Series:
        """Detect anomalies in volatility term structure"""
        short_vol = vol_structure['short_term']
        medium_vol = vol_structure['medium_term']
        long_vol = vol_structure['long_term']
        
        # Calculate spreads
        short_medium_spread = short_vol - medium_vol
        medium_long_spread = medium_vol - long_vol
        
        # Anomaly detection
        anomaly_signals = pd.Series(0, index=short_vol.index)
        
        # Inverted term structure (short > long by significant margin)
        inverted_condition = (short_medium_spread > self.vol_spread_threshold) & \
                           (medium_long_spread > self.vol_spread_threshold)
        anomaly_signals[inverted_condition] = -1
        
        # Extremely flat term structure
        flat_condition = (abs(short_medium_spread) < self.vol_spread_threshold / 2) & \
                        (abs(medium_long_spread) < self.vol_spread_threshold / 2)
        anomaly_signals[flat_condition] = 1
        
        return anomaly_signals
    
    def generate_signal(self, market_data: pd.DataFrame) -> str:
        """Generate volatility surface arbitrage signal"""
        if len(market_data) < self.long_term_window * 2:
            return 'HOLD'
        
        vol_structure = self.calculate_term_structure(market_data)
        anomaly_signals = self.detect_term_structure_anomalies(vol_structure)
        
        current_signal = anomaly_signals.iloc[-1]
        
        if current_signal == 1:
            return 'BUY'   # Flat term structure - expect volatility increase
        elif current_signal == -1:
            return 'SELL'  # Inverted term structure - expect normalization
        else:
            return 'HOLD'


class AdaptiveVolatilityAgent(BaseAgent):
    """
    Adaptive volatility strategy that adjusts to changing market regimes
    using multiple volatility measures and regime detection.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.regime_window = self.config.get("regime_window", 60)
        self.vol_threshold_low = self.config.get("vol_threshold_low", 0.15)
        self.vol_threshold_high = self.config.get("vol_threshold_high", 0.35)
        self.regime_change_threshold = self.config.get("regime_change_threshold", 0.1)
        
    def detect_volatility_regime(self, market_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect current volatility regime"""
        returns = market_data['close'].pct_change()
        
        # Rolling volatility
        rolling_vol = returns.rolling(self.regime_window).std() * np.sqrt(252)
        
        # Regime classification
        regime = pd.Series(0, index=returns.index)  # 0: Normal, 1: High Vol, -1: Low Vol
        
        regime[rolling_vol > self.vol_threshold_high] = 1   # High volatility regime
        regime[rolling_vol < self.vol_threshold_low] = -1   # Low volatility regime
        
        # Regime changes
        regime_changes = regime.diff().abs() > 0
        
        return regime, regime_changes
    
    def calculate_regime_persistence(self, regime: pd.Series) -> pd.Series:
        """Calculate how long current regime has persisted"""
        regime_persistence = pd.Series(0, index=regime.index)
        
        current_regime = None
        persistence_count = 0
        
        for i, reg in enumerate(regime):
            if reg != current_regime:
                current_regime = reg
                persistence_count = 1
            else:
                persistence_count += 1
            
            regime_persistence.iloc[i] = persistence_count
        
        return regime_persistence
    
    def generate_signal(self, market_data: pd.DataFrame) -> str:
        """Generate adaptive volatility signal"""
        if len(market_data) < self.regime_window * 2:
            return 'HOLD'
        
        regime, regime_changes = self.detect_volatility_regime(market_data)
        regime_persistence = self.calculate_regime_persistence(regime)
        
        current_regime = regime.iloc[-1]
        current_persistence = regime_persistence.iloc[-1]
        recent_change = regime_changes.iloc[-5:].any()  # Any change in last 5 periods
        
        # Adaptive strategy based on regime
        if current_regime == 1:  # High volatility regime
            if current_persistence > 10:  # Persistent high vol
                return 'SELL'  # Expect mean reversion
            else:
                return 'HOLD'  # Wait for regime to establish
                
        elif current_regime == -1:  # Low volatility regime
            if current_persistence > 20:  # Very persistent low vol
                return 'BUY'   # Expect volatility expansion
            else:
                return 'HOLD'
                
        else:  # Normal regime
            if recent_change:
                return 'HOLD'  # Wait for regime to stabilize
            else:
                return 'HOLD'  # No clear signal in normal regime


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data with volatility clustering
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create GARCH-like volatility clustering
    n = len(dates)
    returns = np.zeros(n)
    volatility = np.zeros(n)
    volatility[0] = 0.02
    
    # GARCH(1,1) parameters
    omega = 0.00001
    alpha = 0.05
    beta = 0.9
    
    for i in range(1, n):
        # GARCH volatility update
        volatility[i] = np.sqrt(omega + alpha * returns[i-1]**2 + beta * volatility[i-1]**2)
        
        # Generate return with current volatility
        returns[i] = volatility[i] * np.random.randn()
    
    # Convert to prices
    log_prices = np.cumsum(returns)
    prices = 100 * np.exp(log_prices)
    
    # Create OHLC data (simplified)
    high_prices = prices * (1 + np.abs(np.random.randn(n)) * 0.01)
    low_prices = prices * (1 - np.abs(np.random.randn(n)) * 0.01)
    open_prices = prices * (1 + np.random.randn(n) * 0.005)
    
    sample_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Test volatility agents
    print("Testing Volatility Trading Agents:")
    print("=" * 50)
    
    # Volatility Breakout Agent
    breakout_agent = VolatilityBreakoutAgent({
        'vol_lookback': 20,
        'breakout_threshold': 2.0,
        'vol_estimation_method': 'garman_klass'
    })
    
    breakout_signals = []
    for i in range(60, len(sample_data)):  # Start after warmup period
        signal = breakout_agent.generate_signal(sample_data.iloc[:i+1])
        breakout_signals.append(signal)
    
    print(f"Volatility Breakout Agent:")
    print(f"  Buy signals: {breakout_signals.count('BUY')}")
    print(f"  Sell signals: {breakout_signals.count('SELL')}")
    print(f"  Hold signals: {breakout_signals.count('HOLD')}")
    
    # VIX-based Agent
    vix_agent = VIXBasedAgent({
        'vix_window': 20,
        'vix_high_threshold': 25,
        'vix_low_threshold': 12
    })
    
    vix_signals = []
    for i in range(40, len(sample_data)):
        signal = vix_agent.generate_signal(sample_data.iloc[:i+1])
        vix_signals.append(signal)
    
    print(f"\nVIX-based Agent:")
    print(f"  Buy signals: {vix_signals.count('BUY')}")
    print(f"  Sell signals: {vix_signals.count('SELL')}")
    print(f"  Hold signals: {vix_signals.count('HOLD')}")
    
    # Adaptive Volatility Agent
    adaptive_agent = AdaptiveVolatilityAgent({
        'regime_window': 30,
        'vol_threshold_low': 0.15,
        'vol_threshold_high': 0.30
    })
    
    adaptive_signals = []
    for i in range(120, len(sample_data)):
        signal = adaptive_agent.generate_signal(sample_data.iloc[:i+1])
        adaptive_signals.append(signal)
    
    print(f"\nAdaptive Volatility Agent:")
    print(f"  Buy signals: {adaptive_signals.count('BUY')}")
    print(f"  Sell signals: {adaptive_signals.count('SELL')}")
    print(f"  Hold signals: {adaptive_signals.count('HOLD')}")