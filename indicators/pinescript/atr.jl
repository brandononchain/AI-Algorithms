//@version=6
indicator("Average True Range (ATR)", overlay=false)

length = input.int(14, title="ATR Length")
atr    = ta.atr(length)

plot(atr, title="ATR", linewidth=2)
