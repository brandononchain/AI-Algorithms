//@version=6
indicator("MACD (Moving Average Convergence Divergence)", overlay=false)

fastLength   = input.int(12, title="Fast EMA Length")
slowLength   = input.int(26, title="Slow EMA Length")
signalLength = input.int(9,  title="Signal EMA Length")

macdLine   = ta.ema(close, fastLength) - ta.ema(close, slowLength)
signalLine = ta.ema(macdLine, signalLength)
histogram  = macdLine - signalLine

plot(macdLine,   title="MACD Line",   linewidth=2)
plot(signalLine, title="Signal Line", linewidth=1)
plot(histogram,  title="Histogram",   style=plot.style_histogram)
