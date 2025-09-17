"""
Data Loading and Preprocessing System

Comprehensive data management for trading algorithms:
- Multiple data source integration
- Data cleaning and validation
- Feature engineering
- Market data normalization
- Real-time and historical data handling
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import logging
from pathlib import Path
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import pickle

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    # Data sources
    primary_source: str = 'yfinance'  # 'yfinance', 'alpha_vantage', 'twelvedata', 'quandl'
    backup_sources: List[str] = field(default_factory=lambda: ['yfinance'])
    
    # Time settings
    start_date: str = '2020-01-01'
    end_date: str = 'today'
    frequency: str = 'daily'  # 'minute', 'hourly', 'daily', 'weekly', 'monthly'
    
    # Data validation
    min_data_points: int = 252  # Minimum required data points
    max_missing_pct: float = 0.05  # Maximum allowed missing data percentage
    outlier_detection: bool = True
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
    
    # Feature engineering
    add_technical_indicators: bool = True
    add_market_features: bool = True
    add_calendar_features: bool = True
    
    # Storage
    cache_data: bool = True
    cache_directory: str = './data_cache'
    database_path: str = './market_data.db'
    
    # API keys (set these as environment variables)
    alpha_vantage_key: Optional[str] = None
    twelvedata_key: Optional[str] = None
    quandl_key: Optional[str] = None


class DataValidator:
    """Data validation and cleaning utilities"""
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate OHLCV data integrity"""
        issues = []
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
            return False, issues
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                issues.append(f"Non-positive values found in {col}")
        
        # Check OHLC relationships
        if (df['high'] < df['low']).any():
            issues.append("High prices lower than low prices")
        
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            issues.append("High prices lower than open/close prices")
        
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            issues.append("Low prices higher than open/close prices")
        
        # Check for excessive missing data
        missing_pct = df.isnull().sum() / len(df)
        excessive_missing = missing_pct[missing_pct > 0.1]
        if not excessive_missing.empty:
            issues.append(f"Excessive missing data: {excessive_missing.to_dict()}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'iqr', 
                       columns: List[str] = None) -> pd.DataFrame:
        """Detect outliers in data"""
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
        
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = z_scores > 3
                
            elif method == 'isolation_forest':
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers[col] = iso_forest.fit_predict(df[[col]].fillna(df[col].mean())) == -1
                except ImportError:
                    logger.warning("scikit-learn not available, falling back to IQR method")
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        return outliers
    
    @staticmethod
    def clean_data(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Clean and preprocess data"""
        df_clean = df.copy()
        
        # Handle missing values
        # Forward fill first, then backward fill
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Detect and handle outliers
        if config.outlier_detection:
            outliers = DataValidator.detect_outliers(df_clean, config.outlier_method)
            
            # Replace outliers with interpolated values
            for col in outliers.columns:
                if col in df_clean.columns:
                    outlier_mask = outliers[col]
                    if outlier_mask.any():
                        df_clean.loc[outlier_mask, col] = np.nan
                        df_clean[col] = df_clean[col].interpolate(method='linear')
        
        # Remove rows with excessive missing data
        missing_pct = df_clean.isnull().sum(axis=1) / len(df_clean.columns)
        df_clean = df_clean[missing_pct <= config.max_missing_pct]
        
        return df_clean


class FeatureEngineer:
    """Feature engineering for trading data"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators"""
        df_features = df.copy()
        
        # Price-based features
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df_features[f'sma_{window}'] = df_features['close'].rolling(window).mean()
            df_features[f'ema_{window}'] = df_features['close'].ewm(span=window).mean()
        
        # Volatility measures
        df_features['volatility_20'] = df_features['returns'].rolling(20).std() * np.sqrt(252)
        df_features['atr_14'] = FeatureEngineer._calculate_atr(df_features, 14)
        
        # Momentum indicators
        df_features['rsi_14'] = FeatureEngineer._calculate_rsi(df_features['close'], 14)
        df_features['momentum_10'] = df_features['close'] / df_features['close'].shift(10) - 1
        
        # MACD
        macd_line, macd_signal, macd_histogram = FeatureEngineer._calculate_macd(df_features['close'])
        df_features['macd'] = macd_line
        df_features['macd_signal'] = macd_signal
        df_features['macd_histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = FeatureEngineer._calculate_bollinger_bands(df_features['close'])
        df_features['bb_upper'] = bb_upper
        df_features['bb_middle'] = bb_middle
        df_features['bb_lower'] = bb_lower
        df_features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df_features['bb_position'] = (df_features['close'] - bb_lower) / (bb_upper - bb_lower)
        
        return df_features
    
    @staticmethod
    def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features"""
        df_features = df.copy()
        
        # Price action features
        df_features['body_size'] = abs(df_features['close'] - df_features['open'])
        df_features['upper_shadow'] = df_features['high'] - np.maximum(df_features['open'], df_features['close'])
        df_features['lower_shadow'] = np.minimum(df_features['open'], df_features['close']) - df_features['low']
        df_features['total_range'] = df_features['high'] - df_features['low']
        
        # Volume features (if available)
        if 'volume' in df_features.columns:
            df_features['volume_sma_20'] = df_features['volume'].rolling(20).mean()
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma_20']
            df_features['price_volume'] = df_features['close'] * df_features['volume']
            df_features['vwap'] = df_features['price_volume'].rolling(20).sum() / df_features['volume'].rolling(20).sum()
        
        # Gap analysis
        df_features['gap'] = (df_features['open'] - df_features['close'].shift(1)) / df_features['close'].shift(1)
        df_features['gap_filled'] = np.where(
            df_features['gap'] > 0,
            df_features['low'] <= df_features['close'].shift(1),
            df_features['high'] >= df_features['close'].shift(1)
        )
        
        return df_features
    
    @staticmethod
    def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features"""
        df_features = df.copy()
        
        # Time-based features
        df_features['year'] = df_features.index.year
        df_features['month'] = df_features.index.month
        df_features['day'] = df_features.index.day
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['day_of_year'] = df_features.index.dayofyear
        df_features['week_of_year'] = df_features.index.isocalendar().week
        
        # Market session features
        df_features['is_monday'] = (df_features['day_of_week'] == 0).astype(int)
        df_features['is_friday'] = (df_features['day_of_week'] == 4).astype(int)
        df_features['is_month_end'] = df_features.index.is_month_end.astype(int)
        df_features['is_month_start'] = df_features.index.is_month_start.astype(int)
        df_features['is_quarter_end'] = df_features.index.is_quarter_end.astype(int)
        
        # Seasonal patterns
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        return df_features
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift())
        low_close_prev = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower


class DataLoader:
    """Main data loading and management class"""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.cache_dir = Path(self.config.cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for data storage"""
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_metadata (
                    symbol TEXT PRIMARY KEY,
                    last_updated TIMESTAMP,
                    source TEXT,
                    data_points INTEGER,
                    start_date TEXT,
                    end_date TEXT
                )
            ''')
    
    def get_data(self, symbols: Union[str, List[str]], 
                 start_date: str = None, end_date: str = None,
                 source: str = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get market data for one or multiple symbols
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            source: Data source to use
            
        Returns:
            DataFrame for single symbol, dict of DataFrames for multiple symbols
        """
        if isinstance(symbols, str):
            return self._get_single_symbol_data(symbols, start_date, end_date, source)
        else:
            return self._get_multiple_symbols_data(symbols, start_date, end_date, source)
    
    def _get_single_symbol_data(self, symbol: str, start_date: str = None, 
                               end_date: str = None, source: str = None) -> pd.DataFrame:
        """Get data for a single symbol"""
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        source = source or self.config.primary_source
        
        if end_date == 'today':
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache first
        if self.config.cache_data:
            cached_data = self._get_cached_data(symbol, start_date, end_date)
            if cached_data is not None and len(cached_data) >= self.config.min_data_points:
                logger.info(f"Using cached data for {symbol}")
                return self._process_data(cached_data, symbol)
        
        # Fetch new data
        logger.info(f"Fetching data for {symbol} from {source}")
        raw_data = self._fetch_data(symbol, start_date, end_date, source)
        
        if raw_data is None or raw_data.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return pd.DataFrame()
        
        # Validate and clean data
        is_valid, issues = self.validator.validate_ohlcv_data(raw_data)
        if not is_valid:
            logger.warning(f"Data validation issues for {symbol}: {issues}")
        
        cleaned_data = self.validator.clean_data(raw_data, self.config)
        
        # Cache the data
        if self.config.cache_data:
            self._cache_data(symbol, cleaned_data, source)
        
        # Process and return
        return self._process_data(cleaned_data, symbol)
    
    def _get_multiple_symbols_data(self, symbols: List[str], start_date: str = None, 
                                  end_date: str = None, source: str = None) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols"""
        results = {}
        
        # Use ThreadPoolExecutor for parallel data fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self._get_single_symbol_data, symbol, start_date, end_date, source): symbol
                for symbol in symbols
            }
            
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        results[symbol] = data
                    else:
                        logger.warning(f"No data retrieved for {symbol}")
                except Exception as exc:
                    logger.error(f"Error fetching data for {symbol}: {exc}")
        
        return results
    
    def _fetch_data(self, symbol: str, start_date: str, end_date: str, source: str) -> pd.DataFrame:
        """Fetch data from specified source"""
        try:
            if source == 'yfinance':
                return self._fetch_yfinance_data(symbol, start_date, end_date)
            elif source == 'alpha_vantage':
                return self._fetch_alpha_vantage_data(symbol, start_date, end_date)
            elif source == 'twelvedata':
                return self._fetch_twelvedata_data(symbol, start_date, end_date)
            else:
                logger.error(f"Unsupported data source: {source}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data from {source}: {e}")
            
            # Try backup sources
            for backup_source in self.config.backup_sources:
                if backup_source != source:
                    logger.info(f"Trying backup source: {backup_source}")
                    try:
                        return self._fetch_data(symbol, start_date, end_date, backup_source)
                    except Exception as backup_e:
                        logger.error(f"Backup source {backup_source} also failed: {backup_e}")
            
            return pd.DataFrame()
    
    def _fetch_yfinance_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            data.index.name = 'date'
            
            return data
        except Exception as e:
            logger.error(f"Error fetching from yfinance: {e}")
            return pd.DataFrame()
    
    def _fetch_alpha_vantage_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage"""
        if not self.config.alpha_vantage_key:
            logger.error("Alpha Vantage API key not provided")
            return pd.DataFrame()
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.config.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                logger.error(f"No data returned from Alpha Vantage for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            df.index.name = 'date'
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            return df
        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {e}")
            return pd.DataFrame()
    
    def _fetch_twelvedata_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Twelve Data"""
        if not self.config.twelvedata_key:
            logger.error("Twelve Data API key not provided")
            return pd.DataFrame()
        
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': '1day',
                'start_date': start_date,
                'end_date': end_date,
                'apikey': self.config.twelvedata_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'values' not in data:
                logger.error(f"No data returned from Twelve Data for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            df.index.name = 'date'
            df.sort_index(inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching from Twelve Data: {e}")
            return pd.DataFrame()
    
    def _get_cached_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get cached data from database"""
        try:
            with sqlite3.connect(self.config.database_path) as conn:
                query = '''
                    SELECT date, open, high, low, close, volume
                    FROM market_data
                    WHERE symbol = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                '''
                df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
                
                if df.empty:
                    return None
                
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
        except Exception as e:
            logger.error(f"Error reading cached data: {e}")
            return None
    
    def _cache_data(self, symbol: str, data: pd.DataFrame, source: str):
        """Cache data to database"""
        try:
            with sqlite3.connect(self.config.database_path) as conn:
                # Prepare data for insertion
                data_to_insert = data.copy()
                data_to_insert['symbol'] = symbol
                data_to_insert['source'] = source
                data_to_insert.reset_index(inplace=True)
                data_to_insert['date'] = data_to_insert['date'].dt.strftime('%Y-%m-%d')
                
                # Insert data
                data_to_insert.to_sql('market_data', conn, if_exists='replace', index=False)
                
                # Update metadata
                metadata = {
                    'symbol': symbol,
                    'last_updated': datetime.now().isoformat(),
                    'source': source,
                    'data_points': len(data),
                    'start_date': data.index.min().strftime('%Y-%m-%d'),
                    'end_date': data.index.max().strftime('%Y-%m-%d')
                }
                
                conn.execute('''
                    INSERT OR REPLACE INTO data_metadata 
                    (symbol, last_updated, source, data_points, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', tuple(metadata.values()))
                
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def _process_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process raw data with feature engineering"""
        processed_data = data.copy()
        
        # Add technical indicators
        if self.config.add_technical_indicators:
            processed_data = self.feature_engineer.add_technical_indicators(processed_data)
        
        # Add market features
        if self.config.add_market_features:
            processed_data = self.feature_engineer.add_market_features(processed_data)
        
        # Add calendar features
        if self.config.add_calendar_features:
            processed_data = self.feature_engineer.add_calendar_features(processed_data)
        
        return processed_data
    
    def get_data_info(self, symbol: str = None) -> pd.DataFrame:
        """Get information about cached data"""
        try:
            with sqlite3.connect(self.config.database_path) as conn:
                if symbol:
                    query = "SELECT * FROM data_metadata WHERE symbol = ?"
                    params = (symbol,)
                else:
                    query = "SELECT * FROM data_metadata"
                    params = ()
                
                df = pd.read_sql_query(query, conn, params=params)
                return df
        except Exception as e:
            logger.error(f"Error getting data info: {e}")
            return pd.DataFrame()
    
    def clear_cache(self, symbol: str = None):
        """Clear cached data"""
        try:
            with sqlite3.connect(self.config.database_path) as conn:
                if symbol:
                    conn.execute("DELETE FROM market_data WHERE symbol = ?", (symbol,))
                    conn.execute("DELETE FROM data_metadata WHERE symbol = ?", (symbol,))
                else:
                    conn.execute("DELETE FROM market_data")
                    conn.execute("DELETE FROM data_metadata")
                
                logger.info(f"Cache cleared for {'all symbols' if not symbol else symbol}")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


# Example usage
if __name__ == "__main__":
    # Create data loader with configuration
    config = DataConfig(
        start_date='2020-01-01',
        end_date='2023-12-31',
        add_technical_indicators=True,
        add_market_features=True,
        cache_data=True
    )
    
    loader = DataLoader(config)
    
    # Load single symbol
    print("Loading data for AAPL...")
    aapl_data = loader.get_data('AAPL')
    print(f"AAPL data shape: {aapl_data.shape}")
    print(f"AAPL columns: {list(aapl_data.columns)}")
    print(f"AAPL date range: {aapl_data.index.min()} to {aapl_data.index.max()}")
    
    # Load multiple symbols
    print("\nLoading data for multiple symbols...")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    multi_data = loader.get_data(symbols)
    
    for symbol, data in multi_data.items():
        print(f"{symbol}: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Show data info
    print("\nCached data info:")
    info = loader.get_data_info()
    print(info)