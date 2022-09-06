import yfinance as yf
import numpy as np

M_data = yf.download("BP SHEL", start="2017-01-01", end="2017-01-30")

def clean(d):
    # The ones column to be added
    col = np.ones(shape = d.T.shape, dtype=int)
    
    X = d["Open"].T
    print(X)

    Y = d["Close"].T
    print(Y)

X_t, Y = clean(M_data)

