# %% [markdown]
"""
# Interactive Dashboard

Build a Plotly Dash app to backtest strategies on arbitrary tickers.
"""

# %% [code]
import dash
from dash import dcc, html
import plotly.express as px

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Input(id="symbol", value="AAPL", type="text"),
    dcc.DatePickerRange(id="daterange", start_date="2025-01-01", end_date="2025-07-20"),
    html.Button("Run", id="run-btn"),
    dcc.Graph(id="equity-curve")
])

@app.callback(
    dash.dependencies.Output("equity-curve", "figure"),
    [dash.dependencies.Input("run-btn", "n_clicks")],
    [dash.dependencies.State("symbol", "value"),
     dash.dependencies.State("daterange", "start_date"),
     dash.dependencies.State("daterange", "end_date")]
)
def update_graph(n, sym, start, end):
    df = fetch_and_prepare(sym, start, end)
    result = VectorBacktester(df).apply_signal(simple_signals(df))
    return px.line(result, x=result.index, y="equity_curve")

if __name__ == "__main__":
    app.run_server(debug=True)
