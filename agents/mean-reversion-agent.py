import pandas as pd
from agents.base_agent import BaseAgent

class MeanReversionAgent(BaseAgent):
    """
    Buys when price deviates below mean by threshold, sells when above.
    """
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.lookback = self.config.get("lookback", 20)
        self.z_threshold = self.config.get("z_threshold", 1.5)

    def generate_signal(self, market_data: pd.DataFrame) -> str:
        window = market_data['close'].rolling(self.lookback)
        mean = window.mean().iloc[-1]
        std = window.std().iloc[-1]
        latest = market_data['close'].iloc[-1]
        z_score = (latest - mean) / std if std else 0
        if z_score > self.z_threshold:
            return 'SELL'
        elif z_score < -self.z_threshold:
            return 'BUY'
        return 'HOLD'
