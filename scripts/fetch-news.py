import os
import requests
import csv
from datetime import datetime

API_KEY = os.getenv("NEWSAPI_KEY")

def fetch_headlines(query: str, page_size: int=100):
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": page_size, "apiKey": API_KEY}
    res = requests.get(url, params=params).json().get("articles", [])
    with open(f"news_{query}_{datetime.utcnow().date()}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["publishedAt","title","description","url"])
        writer.writeheader()
        for art in res:
            writer.writerow({k: art[k] for k in writer.fieldnames})

if __name__ == "__main__":
    fetch_headlines("AAPL OR Apple")
