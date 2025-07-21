//@version=6
indicator("Relative Strength Index (RSI)", overlay=false)

length = input.int(14, title="RSI Length")
src    = input.source(close, title="Source")
rsi    = ta.rsi(src, length)

h1 = hline(70, "Overbought", color=color.red)
h2 = hline(30, "Oversold",   color=color.green)

plot(rsi, title="RSI", linewidth=2)
