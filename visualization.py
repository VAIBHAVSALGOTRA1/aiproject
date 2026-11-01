import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

print("Downloading NIFTY 50 data...")

# Step 1: Download historical data
data = yf.download(
    "^NSEI",
    period="1mo",
    interval="1d",
    auto_adjust=False
)

if data.empty:
    print("No data received. Please try again.")

else:
    # Step 2: Simplify column headings
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Step 3: Remove rows with missing closing prices
    data = data.dropna(subset=["Close"]).copy()

    # Step 4: Create the line chart
    plt.figure(figsize=(12, 6))

    plt.plot(
        data.index,
        data["Close"],
        marker="o",
        label="NIFTY 50 Closing Level"
    )

    # Step 5: Add chart labels
    plt.title("NIFTY 50 Historical Closing Levels")
    plt.xlabel("Date")
    plt.ylabel("Closing Level")

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Step 6: Adjust layout and display
    plt.tight_layout()
    plt.show()

    print("Graph displayed successfully!")