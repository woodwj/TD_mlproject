import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

class pen_inv_model:

    def __init__(self, dataset):
        self.data = dataset

    def train(self):
        self.W = np.linalg.pinv(self.data.X_tr) @ self.data.Y_tr

    def f(self,X_t):
        return  X_t @ self.W

    def error(self):
        return self.data.Y_te - self.f(self.data.X_te)

    def RMSE(self):
        return np.sqrt(np.mean((self.error())**2))

    def MAPE(self):
      return mean_absolute_percentage_error(self.data.Y_te,self.f(self.data.X_te))