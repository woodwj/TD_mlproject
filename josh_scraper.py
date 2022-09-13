import yfinance as yf

import numpy

#data for BP and SHEL 
data = yf.download("BP SHEL", start="2017-01-01", end="2017-01-30")
print(data["Close"])

#difference between closing prices day-to-day
DeltaClosePs = data.diff(axis=0)
print(DeltaClosePs["Close"],"\n")

#ActualClose = yf.download("BP SHEL",start="2017-01-30",end="2017-01-31")
#print("Close prices for 31st Jan were:\n",ActualClose["Close"])

