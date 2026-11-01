import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

print("Downloading NIFTY 50 data...")

# Download historical data
data = yf.download(
    "^NSEI",
    period="6mo",
    interval="1d",
    auto_adjust=False
)

if data.empty:
    print("No data received. Please try again.")

else:
    # Simplify column headings if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Remove missing closing prices
    data = data.dropna(subset=["Close"]).copy()

    # Calculate 20-day and 50-day moving averages
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()

    # Display the latest results
    print("\nLatest NIFTY 50 data with moving averages:")
    print(data[["Close", "SMA_20", "SMA_50"]].tail(10).round(2))

    # Plot closing price and moving averages
    plt.figure(figsize=(12, 6))

    plt.plot(data.index, data["Close"], label="NIFTY 50 Close")
    plt.plot(data.index, data["SMA_20"], label="20-Day SMA")
    plt.plot(data.index, data["SMA_50"], label="50-Day SMA")

    plt.title("NIFTY 50 Closing Level and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Index Level")

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    print("\nMoving average analysis completed!")