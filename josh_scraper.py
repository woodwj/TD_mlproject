import yfinance as yf

data = yf.download("BP SHEL", start="2017-01-01", end="2017-01-30")

print(data["Close"])

DeltaClosePs = data.diff(axis=0)
print(DeltaClosePs["Close"])