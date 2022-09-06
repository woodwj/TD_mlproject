import scraper
import numpy as np

def train(X_t, Y):
    junk = np.linalg.pinv(X_t)
    return junk @ Y

def f(W,X):
    return  X @ W

X_t, Y = scraper.fetch()
W = train(X_t,Y)
Y_pre = f(W,X_t)
print(Y_pre, Y)
error = Y-f(W,X_t) 
print(error)