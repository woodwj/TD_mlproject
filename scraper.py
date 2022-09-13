import yfinance as yf
import numpy as np

class dataset:
  
    def fetch(self, stock, period):
        d = yf.download(stock, period=period)
        return d["Close"].to_numpy().reshape(d.shape[0],1), d["Open"].to_numpy().reshape(d.shape[0],1)

    def tilda(self,data):
        return np.concatenate([np.ones((data.shape[0],1)), data],axis=1)

    # use "period" instead of start/end
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # (optional, default is '1mo')
    def __init__(self,stocks,period):
        X,Y = self.fetch(stocks, period)
        split = int(len(X)*0.8)
        X_tr = X[:split]
        X_te = X[split:]
        self.Y_tr = Y[:split]
        self.Y_te = Y[split:]
        self.X_te = self.tilda(X_te)
        self.X_tr = self.tilda(X_tr)
        self.X_val = self.tilda(X)
        self.Y_val = Y




