import yfinance as yf
import numpy as np

M_data = yf.download("BP SHEL", start="2017-01-01", end="2017-01-30")

def clean(d):
    
    X = d["Open"].to_numpy().T
    print(X)

    Y = d["Close"].to_numpy().T
    print(Y)

X_t, Y = clean(M_data)



