"""
Black-Scholes options pricing and Greeks calculation.
"""
import numpy as np
from scipy.stats import norm
from typing import Dict, Optional
from datetime import datetime
import math

from data_models import OptionContract, OptionType, Greeks, MarketData


class GreeksCalculator:
    """
    Black-Scholes options pricing and Greeks calculator.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def black_scholes_price(
        self, 
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: OptionType,
        dividend_yield: float = 0.0
    ) -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            spot_price: Current price of underlying
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility (annualized)
            option_type: CALL or PUT
            dividend_yield: Dividend yield (annualized)
        
        Returns:
            Theoretical option price
        """
        if time_to_expiry <= 0:
            # At expiry, option worth intrinsic value
            if option_type == OptionType.CALL:
                return max(spot_price - strike, 0)
            else:
                return max(strike - spot_price, 0)
        
        # Adjust for dividends
        adjusted_spot = spot_price * np.exp(-dividend_yield * time_to_expiry)
        
        d1 = (np.log(adjusted_spot / strike) + 
              (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (
              volatility * np.sqrt(time_to_expiry))
        
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type == OptionType.CALL:
            price = (adjusted_spot * norm.cdf(d1) - 
                    strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:  # PUT
            price = (strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                    adjusted_spot * norm.cdf(-d1))
        
        return max(price, 0)  # Price cannot be negative
    
    def calculate_greeks(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: OptionType,
        dividend_yield: float = 0.0
    ) -> Greeks:
        """
        Calculate all Greeks for an option.
        
        Returns:
            Greeks object with delta, gamma, theta, vega, rho
        """
        if time_to_expiry <= 0:
            # At expiry, Greeks are mostly zero except delta
            if option_type == OptionType.CALL:
                delta = 1.0 if spot_price > strike else 0.0
            else:
                delta = -1.0 if spot_price < strike else 0.0
            
            return Greeks(delta=delta, gamma=0, theta=0, vega=0, rho=0)
        
        # Adjust for dividends
        adjusted_spot = spot_price * np.exp(-dividend_yield * time_to_expiry)
        
        d1 = (np.log(adjusted_spot / strike) + 
              (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (
              volatility * np.sqrt(time_to_expiry))
        
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # Standard normal PDF and CDF
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = np.exp(-dividend_yield * time_to_expiry) * cdf_d1
        else:
            delta = -np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
        
        # Gamma (same for calls and puts)
        gamma = (np.exp(-dividend_yield * time_to_expiry) * pdf_d1) / (
                spot_price * volatility * np.sqrt(time_to_expiry))
        
        # Theta
        theta_term1 = -(adjusted_spot * pdf_d1 * volatility) / (2 * np.sqrt(time_to_expiry))
        theta_term2 = self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry)
        theta_term3 = dividend_yield * adjusted_spot
        
        if option_type == OptionType.CALL:
            theta = (theta_term1 - theta_term2 * cdf_d2 + theta_term3 * cdf_d1) / 365
        else:
            theta = (theta_term1 + theta_term2 * norm.cdf(-d2) - theta_term3 * norm.cdf(-d1)) / 365
        
        # Vega (same for calls and puts)
        vega = adjusted_spot * pdf_d1 * np.sqrt(time_to_expiry) / 100  # Per 1% vol change
        
        # Rho
        if option_type == OptionType.CALL:
            rho = strike * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) * cdf_d2 / 100
        else:
            rho = -strike * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
        
        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
    
    def implied_volatility(
        self,
        option_price: float,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        option_type: OptionType,
        dividend_yield: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Returns:
            Implied volatility or None if convergence fails
        """
        if time_to_expiry <= 0:
            return None
        
        # Initial guess
        vol = 0.2
        
        for i in range(max_iterations):
            # Calculate price and vega at current vol
            price = self.black_scholes_price(
                spot_price, strike, time_to_expiry, vol, option_type, dividend_yield
            )
            
            greeks = self.calculate_greeks(
                spot_price, strike, time_to_expiry, vol, option_type, dividend_yield
            )
            
            vega = greeks.vega * 100  # Convert back to per unit vol
            
            if abs(vega) < 1e-10:  # Avoid division by zero
                break
            
            # Newton-Raphson update
            price_diff = price - option_price
            vol_new = vol - price_diff / vega
            
            # Ensure volatility stays positive
            vol_new = max(vol_new, 0.001)
            
            if abs(vol_new - vol) < tolerance:
                return vol_new
            
            vol = vol_new
        
        return None  # Failed to converge
    
    def calculate_option_metrics(
        self,
        contract: OptionContract,
        market_data: MarketData,
        underlying_price: float
    ) -> Dict[str, float]:
        """
        Calculate comprehensive option metrics.
        
        Returns:
            Dictionary with price, Greeks, and other metrics
        """
        time_to_expiry = contract.time_to_expiry
        
        # Use implied volatility if available, otherwise estimate
        if market_data.implied_volatility:
            volatility = market_data.implied_volatility
        else:
            # Estimate volatility from bid-ask spread (rough approximation)
            volatility = max(market_data.spread / market_data.mid_price, 0.1)
        
        # Calculate theoretical price
        theoretical_price = self.black_scholes_price(
            underlying_price, contract.strike, time_to_expiry, 
            volatility, contract.option_type
        )
        
        # Calculate Greeks
        greeks = self.calculate_greeks(
            underlying_price, contract.strike, time_to_expiry,
            volatility, contract.option_type
        )
        
        # Calculate implied volatility from market price
        market_iv = self.implied_volatility(
            market_data.mid_price, underlying_price, contract.strike,
            time_to_expiry, contract.option_type
        )
        
        # Intrinsic and time value
        if contract.option_type == OptionType.CALL:
            intrinsic_value = max(underlying_price - contract.strike, 0)
        else:
            intrinsic_value = max(contract.strike - underlying_price, 0)
        
        time_value = market_data.mid_price - intrinsic_value
        
        return {
            'theoretical_price': theoretical_price,
            'market_price': market_data.mid_price,
            'bid': market_data.bid,
            'ask': market_data.ask,
            'delta': greeks.delta,
            'gamma': greeks.gamma,
            'theta': greeks.theta,
            'vega': greeks.vega,
            'rho': greeks.rho,
            'implied_volatility': market_iv or volatility,
            'intrinsic_value': intrinsic_value,
            'time_value': time_value,
            'moneyness': underlying_price / contract.strike,
            'time_to_expiry': time_to_expiry
        }
    
    def delta_hedge_ratio(
        self,
        portfolio_delta: float,
        underlying_delta: float = 1.0
    ) -> int:
        """
        Calculate the number of underlying shares needed to hedge delta.
        
        Args:
            portfolio_delta: Current portfolio delta
            underlying_delta: Delta of underlying (usually 1.0)
        
        Returns:
            Number of shares to buy/sell (negative means sell)
        """
        return -int(portfolio_delta / underlying_delta)
    
    def gamma_scalp_signal(
        self,
        current_price: float,
        previous_price: float,
        gamma: float,
        threshold: float = 0.01
    ) -> Optional[str]:
        """
        Generate gamma scalping signal based on price movement.
        
        Returns:
            'buy', 'sell', or None
        """
        price_change = current_price - previous_price
        price_change_pct = price_change / previous_price if previous_price > 0 else 0
        
        if abs(price_change_pct) < threshold:
            return None
        
        # Buy when price goes down (to capture gamma), sell when price goes up
        if gamma > 0:  # Long gamma
            return 'buy' if price_change < 0 else 'sell'
        else:  # Short gamma
            return 'sell' if price_change < 0 else 'buy'