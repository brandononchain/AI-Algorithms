import os
from abc import ABC, abstractmethod
import pandas as pd
import openai

# Ensure your OPENAI_API_KEY is set in .env or environment
openai.api_key = os.getenv("OPENAI_API_KEY")

class BaseAgent(ABC):
    """
    Abstract base class for AI trading agents.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}

    @abstractmethod
    def generate_signal(self, market_data: pd.DataFrame) -> str:
        """
        Analyze market_data and return a trading signal: 'BUY', 'SELL', or 'HOLD'.
        """
        ...

    def execute(self, market_data: pd.DataFrame):
        signal = self.generate_signal(market_data)
        # Placeholder for order execution logic
        print(f"[{self.__class__.__name__}] Signal: {signal}")
