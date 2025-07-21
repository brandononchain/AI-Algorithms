import asyncio
import pandas as pd
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager

async def save_orderbook(symbol: str, limit: int=100):
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    ts = bm.depth_socket(symbol.lower(), limit=limit)
    async with ts as stream:
        msg = await stream.recv()
        bids = pd.DataFrame(msg["bids"], columns=["price","qty"], dtype=float)
        asks = pd.DataFrame(msg["asks"], columns=["price","qty"], dtype=float)
        snapshot = {
            "timestamp": datetime.utcfromtimestamp(msg["E"]/1000),
            "bids": bids, "asks": asks
        }
        # Save to parquet
        pd.concat([bids, asks], axis=1).to_parquet(f"orderbook_{symbol}_{msg['E']}.parquet")
    await client.close_connection()

if __name__ == "__main__":
    asyncio.run(save_orderbook("BTCUSDT"))
