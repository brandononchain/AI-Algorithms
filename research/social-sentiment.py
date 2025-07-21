# %% [markdown]
"""
# Alternative Data: Social Media Sentiment

Fetch Reddit/Twitter posts, classify with LLM, and compare to price signals.
"""

# %% [code]
import pandas as pd
from praw import Reddit
import openai

# %% [code]
redd = Reddit(client_id=os.getenv("REDDIT_ID"), client_secret=os.getenv("REDDIT_SECRET"), user_agent="ai-algo")
posts = redd.subreddit("stocks").hot(limit=100)
df_posts = pd.DataFrame([{"text": p.title+p.selftext} for p in posts])
df_posts["sentiment"] = df_posts["text"].apply(lambda t: openai.Completion.create(
    engine="text-davinci-003",
    prompt=f"Sentiment of: {t}",
    max_tokens=5
).choices[0].text.strip())
