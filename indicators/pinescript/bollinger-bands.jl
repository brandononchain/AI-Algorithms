//@version=6
indicator("Bollinger Bands", overlay=true)

length = input.int(20, title="Length")
mult   = input.float(2.0, title="StdDev Multiplier")
src     = input.source(close, title="Source")

basis = ta.sma(src, length)
dev   = mult * ta.stdev(src, length)

upper = basis + dev
lower = basis - dev

p_base = plot(basis, title="Middle Band", linewidth=1)
p_up   = plot(upper,  title="Upper Band",  linewidth=1)
p_low  = plot(lower,  title="Lower Band",  linewidth=1)

fill(p_up, p_low, color=color.new(color.gray, 90))
