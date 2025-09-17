"""
Momentum Trading Agent

Multi-timeframe momentum strategy that captures trending moves
using various momentum indicators and filters.
"""

import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent
from typing import Dict, List, Optional, Tuple
import talib


class MomentumAgent(BaseAgent):
    """
    Momentum trading strategy using multiple timeframes and indicators:
    - Price momentum (rate of change)
    - RSI momentum
    - MACD momentum
    - Volume confirmation
    - Trend strength filters
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        
        # Momentum parameters
        self.fast_period = self.config.get("fast_period", 10)
        self.slow_period = self.config.get("slow_period", 20)
        self.momentum_threshold = self.config.get("momentum_threshold", 0.02)
        
        # RSI parameters
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        
        # MACD parameters
        self.macd_fast = self.config.get("macd_fast", 12)
        self.macd_slow = self.config.get("macd_slow", 26)
        self.macd_signal = self.config.get("macd_signal", 9)
        
        # Volume parameters
        self.volume_ma_period = self.config.get("volume_ma_period", 20)
        self.volume_threshold = self.config.get("volume_threshold", 1.2)
        
        # Risk management
        self.min_trend_strength = self.config.get("min_trend_strength", 0.5)
        self.max_volatility = self.config.get("max_volatility", 0.05)
    
    def calculate_price_momentum(self, prices: pd.Series) -> pd.Series:
        """Calculate price momentum (rate of change)"""
        return prices.pct_change(self.fast_period)
    
    def calculate_momentum_strength(self, prices: pd.Series) -> pd.Series:
        """Calculate momentum strength using multiple periods"""
        mom_fast = prices.pct_change(self.fast_period)
        mom_slow = prices.pct_change(self.slow_period)
        
        # Momentum strength is the ratio of fast to slow momentum
        momentum_strength = mom_fast / (mom_slow + 1e-8)  # Add small value to avoid division by zero
        return momentum_strength
    
    def calculate_rsi_momentum(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI-based momentum signals"""
        try:
            rsi = talib.RSI(prices.values, timeperiod=self.rsi_period)
            rsi_series = pd.Series(rsi, index=prices.index)
            
            # RSI momentum: positive when RSI is rising and above 50
            rsi_change = rsi_series.diff()
            rsi_momentum = np.where(
                (rsi_series > 50) & (rsi_change > 0), 1,
                np.where((rsi_series < 50) & (rsi_change < 0), -1, 0)
            )
            
            return pd.Series(rsi_momentum, index=prices.index)
        except:
            # Fallback manual RSI calculation
            return self._manual_rsi_momentum(prices)
    
    def _manual_rsi_momentum(self, prices: pd.Series) -> pd.Series:
        """Manual RSI calculation as fallback"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi_change = rsi.diff()
        rsi_momentum = np.where(
            (rsi > 50) & (rsi_change > 0), 1,
            np.where((rsi < 50) & (rsi_change < 0), -1, 0)
        )
        
        return pd.Series(rsi_momentum, index=prices.index)
    
    def calculate_macd_momentum(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD-based momentum"""
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                prices.values, 
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            
            macd_series = pd.Series(macd, index=prices.index)
            signal_series = pd.Series(macd_signal, index=prices.index)
            
            # MACD momentum: positive when MACD > signal and both rising
            macd_momentum = np.where(
                (macd_series > signal_series) & (macd_series.diff() > 0), 1,
                np.where((macd_series < signal_series) & (macd_series.diff() < 0), -1, 0)
            )
            
            return pd.Series(macd_momentum, index=prices.index)
        except:
            return self._manual_macd_momentum(prices)
    
    def _manual_macd_momentum(self, prices: pd.Series) -> pd.Series:
        """Manual MACD calculation as fallback"""
        ema_fast = prices.ewm(span=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.macd_signal).mean()
        
        macd_momentum = np.where(
            (macd > signal) & (macd.diff() > 0), 1,
            np.where((macd < signal) & (macd.diff() < 0), -1, 0)
        )
        
        return pd.Series(macd_momentum, index=prices.index)
    
    def calculate_volume_confirmation(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate volume-based confirmation"""
        if 'volume' not in market_data.columns:
            return pd.Series(1, index=market_data.index)  # No volume data
        
        volume = market_data['volume']
        volume_ma = volume.rolling(self.volume_ma_period).mean()
        
        # Volume confirmation: 1 if above average, 0 otherwise
        volume_conf = (volume > volume_ma * self.volume_threshold).astype(int)
        return volume_conf
    
    def calculate_trend_strength(self, prices: pd.Series) -> pd.Series:
        """Calculate trend strength using ADX-like measure"""
        high = prices  # Simplified - using close as high
        low = prices   # Simplified - using close as low
        close = prices
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                          np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                           np.maximum(low.shift(1) - low, 0), 0)
        
        dm_plus = pd.Series(dm_plus, index=prices.index)
        dm_minus = pd.Series(dm_minus, index=prices.index)
        
        # Smooth the values
        period = 14
        tr_smooth = true_range.rolling(period).mean()
        dm_plus_smooth = dm_plus.rolling(period).mean()
        dm_minus_smooth = dm_minus.rolling(period).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # Calculate DX and ADX (trend strength)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)
        adx = dx.rolling(period).mean()
        
        return adx / 100  # Normalize to 0-1 range
    
    def calculate_volatility_filter(self, prices: pd.Series) -> pd.Series:
        """Calculate volatility filter to avoid trading in high volatility periods"""
        returns = prices.pct_change()
        volatility = returns.rolling(20).std()
        
        # Filter: 1 if volatility is acceptable, 0 otherwise
        vol_filter = (volatility < self.max_volatility).astype(int)
        return vol_filter
    
    def generate_signal(self, market_data: pd.DataFrame) -> str:
        """Generate momentum trading signal"""
        prices = market_data['close']
        
        if len(prices) < max(self.slow_period, self.rsi_period, 30):
            return 'HOLD'
        
        # Calculate all momentum indicators
        price_momentum = self.calculate_price_momentum(prices)
        momentum_strength = self.calculate_momentum_strength(prices)
        rsi_momentum = self.calculate_rsi_momentum(prices)
        macd_momentum = self.calculate_macd_momentum(prices)
        volume_conf = self.calculate_volume_confirmation(market_data)
        trend_strength = self.calculate_trend_strength(prices)
        vol_filter = self.calculate_volatility_filter(prices)
        
        # Get latest values
        current_price_mom = price_momentum.iloc[-1]
        current_mom_strength = momentum_strength.iloc[-1]
        current_rsi_mom = rsi_momentum.iloc[-1]
        current_macd_mom = macd_momentum.iloc[-1]
        current_volume_conf = volume_conf.iloc[-1]
        current_trend_strength = trend_strength.iloc[-1]
        current_vol_filter = vol_filter.iloc[-1]
        
        # Skip if conditions are not met
        if (current_vol_filter == 0 or 
            current_trend_strength < self.min_trend_strength or
            current_volume_conf == 0):
            return 'HOLD'
        
        # Combine momentum signals
        momentum_score = 0
        
        # Price momentum (strongest weight)
        if abs(current_price_mom) > self.momentum_threshold:
            momentum_score += 3 * np.sign(current_price_mom)
        
        # Momentum strength
        if abs(current_mom_strength) > 1.2:
            momentum_score += 2 * np.sign(current_mom_strength)
        
        # Technical momentum indicators
        momentum_score += current_rsi_mom
        momentum_score += current_macd_mom
        
        # Weight by trend strength
        momentum_score *= current_trend_strength
        
        # Generate final signal
        if momentum_score > 2:
            return 'BUY'
        elif momentum_score < -2:
            return 'SELL'
        else:
            return 'HOLD'
    
    def generate_detailed_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate detailed momentum signals with all indicators"""
        prices = market_data['close']
        
        # Calculate all indicators
        price_momentum = self.calculate_price_momentum(prices)
        momentum_strength = self.calculate_momentum_strength(prices)
        rsi_momentum = self.calculate_rsi_momentum(prices)
        macd_momentum = self.calculate_macd_momentum(prices)
        volume_conf = self.calculate_volume_confirmation(market_data)
        trend_strength = self.calculate_trend_strength(prices)
        vol_filter = self.calculate_volatility_filter(prices)
        
        # Combine into signals
        momentum_scores = []
        signals = []
        
        for i in range(len(prices)):
            if i < max(self.slow_period, self.rsi_period, 30):
                momentum_scores.append(0)
                signals.append(0)
                continue
            
            # Get current values
            price_mom = price_momentum.iloc[i]
            mom_strength = momentum_strength.iloc[i]
            rsi_mom = rsi_momentum.iloc[i]
            macd_mom = macd_momentum.iloc[i]
            vol_conf = volume_conf.iloc[i]
            trend_str = trend_strength.iloc[i]
            vol_filt = vol_filter.iloc[i]
            
            # Skip if conditions are not met
            if (vol_filt == 0 or trend_str < self.min_trend_strength or vol_conf == 0):
                momentum_scores.append(0)
                signals.append(0)
                continue
            
            # Calculate momentum score
            momentum_score = 0
            
            if abs(price_mom) > self.momentum_threshold:
                momentum_score += 3 * np.sign(price_mom)
            
            if abs(mom_strength) > 1.2:
                momentum_score += 2 * np.sign(mom_strength)
            
            momentum_score += rsi_mom + macd_mom
            momentum_score *= trend_str
            
            momentum_scores.append(momentum_score)
            
            # Generate signal
            if momentum_score > 2:
                signals.append(1)
            elif momentum_score < -2:
                signals.append(-1)
            else:
                signals.append(0)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'price': prices,
            'price_momentum': price_momentum,
            'momentum_strength': momentum_strength,
            'rsi_momentum': rsi_momentum,
            'macd_momentum': macd_momentum,
            'volume_confirmation': volume_conf,
            'trend_strength': trend_strength,
            'volatility_filter': vol_filter,
            'momentum_score': momentum_scores,
            'signal': signals
        }, index=prices.index)
        
        return results


class VolatilityMomentumAgent(BaseAgent):
    """
    Volatility-adjusted momentum strategy that scales position size
    based on volatility and momentum strength.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.lookback_period = self.config.get("lookback_period", 20)
        self.momentum_threshold = self.config.get("momentum_threshold", 0.01)
        self.vol_lookback = self.config.get("vol_lookback", 20)
        self.target_volatility = self.config.get("target_volatility", 0.15)
    
    def calculate_volatility_adjusted_momentum(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate momentum adjusted for volatility"""
        returns = prices.pct_change()
        
        # Calculate rolling volatility
        volatility = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
        
        # Calculate momentum
        momentum = prices.pct_change(self.lookback_period)
        
        # Volatility-adjusted momentum
        vol_adj_momentum = momentum / (volatility / self.target_volatility)
        
        return vol_adj_momentum, volatility
    
    def generate_signal(self, market_data: pd.DataFrame) -> str:
        """Generate volatility-adjusted momentum signal"""
        prices = market_data['close']
        
        if len(prices) < max(self.lookback_period, self.vol_lookback):
            return 'HOLD'
        
        vol_adj_momentum, volatility = self.calculate_volatility_adjusted_momentum(prices)
        
        current_momentum = vol_adj_momentum.iloc[-1]
        current_volatility = volatility.iloc[-1]
        
        # Avoid trading in extreme volatility conditions
        if current_volatility > 2 * self.target_volatility:
            return 'HOLD'
        
        # Generate signal based on volatility-adjusted momentum
        if current_momentum > self.momentum_threshold:
            return 'BUY'
        elif current_momentum < -self.momentum_threshold:
            return 'SELL'
        else:
            return 'HOLD'


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data with momentum patterns
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create trending price data
    trend = np.linspace(0, 2, len(dates))  # Upward trend
    noise = np.random.randn(len(dates)) * 0.02
    momentum_shocks = np.random.randn(len(dates)) * 0.01
    momentum_shocks[::50] *= 5  # Add occasional momentum shocks
    
    log_prices = trend + np.cumsum(noise + momentum_shocks)
    prices = 100 * np.exp(log_prices)
    volumes = np.random.randint(1000, 10000, len(dates))
    
    sample_data = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Test momentum agent
    momentum_agent = MomentumAgent({
        'fast_period': 10,
        'slow_period': 20,
        'momentum_threshold': 0.02,
        'min_trend_strength': 0.3
    })
    
    # Generate detailed signals
    detailed_results = momentum_agent.generate_detailed_signals(sample_data)
    
    print("Momentum Strategy Results:")
    print(f"Total signals: {(detailed_results['signal'] != 0).sum()}")
    print(f"Buy signals: {(detailed_results['signal'] == 1).sum()}")
    print(f"Sell signals: {(detailed_results['signal'] == -1).sum()}")
    print(f"Average momentum score: {detailed_results['momentum_score'].mean():.3f}")