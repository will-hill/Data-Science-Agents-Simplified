from mcp.server.fastmcp import FastMCP
import yfinance as yf
import pandas as pd
import json

mcp = FastMCP("stock-tools")

def _download(ticker: str, period: str, interval: str):
    print(f"Calling yfinance for data\nticker:{ticker}\nperiod:{period}\ninterval:{interval}")
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        return None

    # Flatten MultiIndex columns if present (yfinance returns MultiIndex for single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]

    # Convert date column to string for JSON serialization
    df[date_col] = df[date_col].astype(str)

    return df, date_col

# ---------- Tool 0: From Tutorial ----------

@mcp.tool()
def add(x: int, y: int) -> int:
    """Check if number is prime
        Args: x number of unknown primacy
        Returns: True is x is prime, otherwise false
    """
    return x + y

# ---------- Tool 1: OHLC from prompt ----------

@mcp.tool()
def get_ohlc(ticker: str, period: str = "1mo") -> str:
    """Get OHLC stock data as a formatted table.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, BTC-USD, ^GSPC)
        period: Time period - one of: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max

    Returns a pandas-style table with Date, Open, High, Low, Close, Volume.
    """
    print(f"\nget_ohlc(ticker={ticker}, period={period})\n")
    interval = "1d"
    out = _download(ticker, period, interval)

    if out is None:
        return f"No data found for {ticker} ({period})"

    df, date_col = out

    # Format as a clean table string
    df_display = df[[date_col, "Open", "High", "Low", "Close", "Volume"]].copy()
    df_display.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # Round prices for readability
    for col in ["Open", "High", "Low", "Close"]:
        df_display[col] = df_display[col].round(2)

    header = f"📈 {ticker} OHLC Data ({period}, {interval})\n"
    return header + df_display.to_string(index=False)

# ---------- Tool 2: Plotly chart from prompt ----------

@mcp.tool()
def plot_ohlc(ticker: str, period: str = "1mo") -> str:
    """Create a candlestick chart. Returns JSON data to reconstruct the Plotly figure.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, BTC-USD, ^GSPC)
        period: Time period - one of: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max
    """
    print(f"\nplot_ohlc(ticker={ticker}, period={period})\n")
    interval = "1d"
    out = _download(ticker, period, interval)

    if out is None:
        return json.dumps({"error": f"No data found for {ticker} ({period})"})

    df, date_col = out

    # Return JSON with plot data for client-side rendering
    plot_data = {
        "type": "candlestick",
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "dates": df[date_col].tolist(),
        "open": df["Open"].tolist(),
        "high": df["High"].tolist(),
        "low": df["Low"].tolist(),
        "close": df["Close"].tolist(),
    }
    return json.dumps(plot_data)

if __name__ == "__main__":
    mcp.run(transport="sse")
