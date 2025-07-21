import pandas as pd
from agents.base_agent import BaseAgent

class TrendFollowingAgent(BaseAgent):
    """
    Uses moving average crossovers to detect trends.
    """
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.short_window = self.config.get("short_window", 50)
        self.long_window = self.config.get("long_window", 200)

    def generate_signal(self, market_data: pd.DataFrame) -> str:
        short_ma = market_data['close'].rolling(self.short_window).mean()
        long_ma = market_data['close'].rolling(self.long_window).mean()
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return 'BUY'
        elif short_ma.iloc[-1] < long_ma.iloc[-1]:
            return 'SELL'
        return 'HOLD'
