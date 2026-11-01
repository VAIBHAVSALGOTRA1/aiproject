
import yfinance as yf

print("Downloading NIFTY 50 data...")

data = yf.download("^NSEI", period="1mo", interval="1d")

if data.empty:
    print("No data received. Please check your internet connection.")
else:
    print("\nLatest NIFTY 50 data:")
    print(data.tail())

    print("\nTotal records:", len(data))
    print("\nData downloaded successfully!")

