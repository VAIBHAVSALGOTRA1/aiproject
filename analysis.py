import yfinance as yf
import pandas as pd

print("Downloading NIFTY 50 data...")

# Step 1: Download one month of historical data
data = yf.download(
    "^NSEI",
    period="1mo",
    interval="1d",
    auto_adjust=False
)

# Step 2: Check whether data was downloaded
if data.empty:
    print("No data received. Please try again.")
else:

    # Step 3: Simplify the column headings
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Step 4: Keep rows with valid closing prices
    data = data.dropna(subset=["Close"]).copy()

    # Step 5: Calculate daily returns as percentages
    data["Daily_Return_%"] = data["Close"].pct_change() * 100

    # Step 6: Calculate the change in closing points
    data["Points_Change"] = data["Close"].diff()

    # Step 7: Display the results
    print("\nNIFTY 50 Market Analysis")
    print(data[["Close", "Points_Change", "Daily_Return_%"]].round(2))

    print("\nAnalysis completed successfully!")