import pandas as pd
import openai
from agents.base_agent import BaseAgent

class NewsSentimentAgent(BaseAgent):
    """
    Uses OpenAI to analyze news headlines or articles for sentiment.
    """
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.prompt_template = (
            "Given the following news headlines, classify the market sentiment as bullish, bearish, or neutral:\n\n{headlines}"
        )
        self.engine = self.config.get('engine', 'text-davinci-003')

    def generate_signal(self, market_data: pd.DataFrame) -> str:
        # Assumes market_data contains a 'headline' column
        headlines = market_data['headline'].tolist()[-5:]
        prompt = self.prompt_template.format(headlines="\n".join(headlines))
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=10,
            temperature=0.0
        )
        sentiment = response.choices[0].text.strip().lower()
        if 'bull' in sentiment:
            return 'BUY'
        elif 'bear' in sentiment:
            return 'SELL'
        return 'HOLD'
