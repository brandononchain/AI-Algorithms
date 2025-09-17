"""
Pairs Trading Agent

Statistical arbitrage strategy that trades on mean-reverting relationships
between correlated assets. Uses cointegration and z-score analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import cointeg
from agents.base_agent import BaseAgent
from typing import Tuple, Dict, Optional


class PairsTradingAgent(BaseAgent):
    """
    Pairs trading strategy based on cointegration and mean reversion.
    
    Strategy:
    1. Identify cointegrated pairs
    2. Calculate z-score of spread
    3. Enter positions when z-score exceeds threshold
    4. Exit when z-score reverts to mean
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.lookback_window = self.config.get("lookback_window", 60)
        self.entry_threshold = self.config.get("entry_threshold", 2.0)
        self.exit_threshold = self.config.get("exit_threshold", 0.5)
        self.stop_loss_threshold = self.config.get("stop_loss_threshold", 3.5)
        self.min_half_life = self.config.get("min_half_life", 1)
        self.max_half_life = self.config.get("max_half_life", 30)
        
        # Store pair relationship data
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None
        self.current_position = 0
        
    def calculate_cointegration(self, y1: pd.Series, y2: pd.Series) -> Tuple[float, float, float]:
        """
        Test for cointegration between two price series.
        
        Returns:
            - cointegration test statistic
            - p-value
            - hedge ratio (beta)
        """
        # Perform Engle-Granger cointegration test
        coint_result = cointeg(y1, y2)
        test_stat = coint_result[0]
        p_value = coint_result[1]
        
        # Calculate hedge ratio using OLS regression
        X = np.column_stack([np.ones(len(y2)), y2])
        beta = np.linalg.lstsq(X, y1, rcond=None)[0]
        hedge_ratio = beta[1]
        
        return test_stat, p_value, hedge_ratio
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion for the spread.
        """
        spread_lag = spread.shift(1)
        spread_diff = spread.diff()
        
        # Remove NaN values
        valid_idx = ~(spread_lag.isna() | spread_diff.isna())
        spread_lag_clean = spread_lag[valid_idx]
        spread_diff_clean = spread_diff[valid_idx]
        
        # Regression: spread_diff = alpha + beta * spread_lag + error
        X = np.column_stack([np.ones(len(spread_lag_clean)), spread_lag_clean])
        try:
            coeffs = np.linalg.lstsq(X, spread_diff_clean, rcond=None)[0]
            beta = coeffs[1]
            
            # Half-life calculation
            if beta < 0:
                half_life = -np.log(2) / beta
            else:
                half_life = np.inf
        except:
            half_life = np.inf
            
        return half_life
    
    def calculate_spread_statistics(self, y1: pd.Series, y2: pd.Series, 
                                  hedge_ratio: float) -> Tuple[pd.Series, float, float]:
        """
        Calculate spread and its statistical properties.
        """
        spread = y1 - hedge_ratio * y2
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        return spread, spread_mean, spread_std
    
    def generate_signals_pair(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for a pair of assets.
        
        Args:
            data1: Price data for first asset
            data2: Price data for second asset
            
        Returns:
            Series with signals: 1 (long spread), -1 (short spread), 0 (no position)
        """
        prices1 = data1['close']
        prices2 = data2['close']
        
        # Ensure same index
        common_index = prices1.index.intersection(prices2.index)
        prices1 = prices1[common_index]
        prices2 = prices2[common_index]
        
        signals = pd.Series(0, index=common_index)
        
        if len(prices1) < self.lookback_window:
            return signals
        
        for i in range(self.lookback_window, len(prices1)):
            # Use rolling window for cointegration analysis
            y1_window = prices1.iloc[i-self.lookback_window:i]
            y2_window = prices2.iloc[i-self.lookback_window:i]
            
            # Test cointegration
            try:
                test_stat, p_value, hedge_ratio = self.calculate_cointegration(y1_window, y2_window)
                
                # Only proceed if pairs are cointegrated (p < 0.05)
                if p_value < 0.05:
                    # Calculate spread
                    spread, spread_mean, spread_std = self.calculate_spread_statistics(
                        y1_window, y2_window, hedge_ratio
                    )
                    
                    # Check half-life
                    half_life = self.calculate_half_life(spread)
                    if not (self.min_half_life <= half_life <= self.max_half_life):
                        continue
                    
                    # Calculate current z-score
                    current_spread = prices1.iloc[i] - hedge_ratio * prices2.iloc[i]
                    z_score = (current_spread - spread_mean) / spread_std
                    
                    # Generate signals based on z-score
                    if abs(z_score) > self.entry_threshold and self.current_position == 0:
                        # Enter position
                        if z_score > 0:
                            signals.iloc[i] = -1  # Short spread (short asset1, long asset2)
                            self.current_position = -1
                        else:
                            signals.iloc[i] = 1   # Long spread (long asset1, short asset2)
                            self.current_position = 1
                    
                    elif abs(z_score) < self.exit_threshold and self.current_position != 0:
                        # Exit position
                        signals.iloc[i] = 0
                        self.current_position = 0
                    
                    elif abs(z_score) > self.stop_loss_threshold and self.current_position != 0:
                        # Stop loss
                        signals.iloc[i] = 0
                        self.current_position = 0
                    
                    else:
                        # Hold current position
                        signals.iloc[i] = self.current_position
                        
            except Exception as e:
                # Skip if cointegration test fails
                continue
        
        return signals
    
    def generate_signal(self, market_data: pd.DataFrame) -> str:
        """
        Generate signal for single asset (not applicable for pairs trading).
        This method is required by base class but pairs trading needs two assets.
        """
        return 'HOLD'
    
    def find_cointegrated_pairs(self, price_data: Dict[str, pd.DataFrame], 
                               min_correlation: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Find cointegrated pairs from a universe of assets.
        
        Args:
            price_data: Dictionary of asset name -> price DataFrame
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of tuples (asset1, asset2, p_value)
        """
        assets = list(price_data.keys())
        cointegrated_pairs = []
        
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                # Get common time period
                prices1 = price_data[asset1]['close']
                prices2 = price_data[asset2]['close']
                common_index = prices1.index.intersection(prices2.index)
                
                if len(common_index) < self.lookback_window:
                    continue
                
                p1 = prices1[common_index]
                p2 = prices2[common_index]
                
                # Check correlation first
                correlation = p1.corr(p2)
                if abs(correlation) < min_correlation:
                    continue
                
                # Test cointegration
                try:
                    test_stat, p_value, hedge_ratio = self.calculate_cointegration(p1, p2)
                    
                    if p_value < 0.05:  # Cointegrated at 5% level
                        cointegrated_pairs.append((asset1, asset2, p_value))
                        
                except Exception:
                    continue
        
        # Sort by p-value (most cointegrated first)
        cointegrated_pairs.sort(key=lambda x: x[2])
        return cointegrated_pairs


class StatisticalArbitrageAgent(BaseAgent):
    """
    Statistical arbitrage strategy using multiple statistical techniques:
    - Mean reversion
    - Momentum
    - Cross-sectional ranking
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.lookback_window = self.config.get("lookback_window", 20)
        self.momentum_window = self.config.get("momentum_window", 10)
        self.reversion_threshold = self.config.get("reversion_threshold", 1.5)
        self.momentum_threshold = self.config.get("momentum_threshold", 0.02)
        
    def calculate_z_score(self, prices: pd.Series, window: int = None) -> pd.Series:
        """Calculate rolling z-score"""
        if window is None:
            window = self.lookback_window
        
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        z_score = (prices - rolling_mean) / rolling_std
        
        return z_score
    
    def calculate_momentum(self, prices: pd.Series, window: int = None) -> pd.Series:
        """Calculate price momentum"""
        if window is None:
            window = self.momentum_window
        
        momentum = prices.pct_change(window)
        return momentum
    
    def generate_signal(self, market_data: pd.DataFrame) -> str:
        """
        Generate trading signal based on statistical arbitrage.
        """
        prices = market_data['close']
        
        if len(prices) < max(self.lookback_window, self.momentum_window):
            return 'HOLD'
        
        # Calculate indicators
        z_score = self.calculate_z_score(prices)
        momentum = self.calculate_momentum(prices)
        
        current_z = z_score.iloc[-1]
        current_momentum = momentum.iloc[-1]
        
        # Mean reversion signal
        reversion_signal = 0
        if current_z > self.reversion_threshold:
            reversion_signal = -1  # Expect reversion down
        elif current_z < -self.reversion_threshold:
            reversion_signal = 1   # Expect reversion up
        
        # Momentum signal
        momentum_signal = 0
        if current_momentum > self.momentum_threshold:
            momentum_signal = 1   # Positive momentum
        elif current_momentum < -self.momentum_threshold:
            momentum_signal = -1  # Negative momentum
        
        # Combine signals (momentum takes precedence for strong moves)
        if abs(current_momentum) > 2 * self.momentum_threshold:
            final_signal = momentum_signal
        else:
            final_signal = reversion_signal
        
        signal_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
        return signal_map[final_signal]


# Example usage
if __name__ == "__main__":
    # Generate sample correlated data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create cointegrated pair
    common_factor = np.cumsum(np.random.randn(len(dates)) * 0.01)
    noise1 = np.random.randn(len(dates)) * 0.005
    noise2 = np.random.randn(len(dates)) * 0.005
    
    prices1 = 100 * np.exp(common_factor + noise1)
    prices2 = 95 * np.exp(0.95 * common_factor + noise2)  # Cointegrated with ratio ~1.05
    
    data1 = pd.DataFrame({'close': prices1}, index=dates)
    data2 = pd.DataFrame({'close': prices2}, index=dates)
    
    # Test pairs trading
    pairs_agent = PairsTradingAgent({
        'lookback_window': 60,
        'entry_threshold': 2.0,
        'exit_threshold': 0.5
    })
    
    signals = pairs_agent.generate_signals_pair(data1, data2)
    print(f"Generated {(signals != 0).sum()} trading signals")
    print(f"Signal distribution: {signals.value_counts()}")