import yfinance as yf
import numpy as np

def clean():
    d = yf.download("BP SHEL", start="2017-01-01", end="2017-01-30")
    # The ones column to be added    
    X = d["Open"].to_numpy().T
    X = np.append(np.ones((X.shape[0],1)), X, axis=1)

    Y = d["Close"].to_numpy().T
    return X,Y

def fetch():
    return clean()




