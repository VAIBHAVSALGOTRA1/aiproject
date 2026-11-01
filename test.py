import yfinance as yf

# Download NIFTY 50 data
data = yf.download("^NSEI", period="1mo", interval="1d")

# Display the data
print("\nComplete Data:")
print(data)

# Display the number of rows
print("\nNumber of rows:")
print(len(data))

# Display only closing prices
print("\nClosing Prices:")
print(data["Close"])

# Display the latest record
print("\nLatest Record:")
print(data.tail(1))