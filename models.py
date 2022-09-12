import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

class model:
    def __init__(self, dataset):
        self.data = dataset

    def f(self,X_t):
        return  X_t @ self.W

    def error(self):
        return self.data.Y_te - self.f(self.data.X_te)

    def RMSE(self):
        return np.sqrt(np.mean((self.error())**2))

    def MAPE(self):
      return mean_absolute_percentage_error(self.data.Y_te,self.f(self.data.X_te))

class pen_inv_model(model):

    def __init__(self, dataset):
        super().__init__(dataset)

    def train(self):
        self.W = np.linalg.pinv(self.data.X_tr) @ self.data.Y_tr

class l2_rls_model(model):

    def __init__(self,dataset):
        super().__init__(dataset)

    def train(self, lmbda):
        """
        lmbda:  positive hyperparameter controls how much regularisation our
                model performs. Allowing us to adjust under or overfitting by 
                using greater or fewer of weights. Tuning this hyperparameter
                allows us to decide how linear should we be between datapoints.

        Returns:    weights that minimises the gradient of our error/objective function.
                    Using penrose inv if l = 0 or the L2 Regularised Least Squares.

                    We use the numpy penrose inverse of our data
                    And if there are infantly many solutions then w has minimum norm.
        """
        X, y = self.data.X_tr, self.data.Y_tr
        
        # Compute the coefficient vector.
        # if lambda is 0 then use pseudo-inverse.
        if lmbda == 0. :
            # libray penrose inverse function (for unified N>d,N<d)
            w = np.linalg.pinv(X) @ y

        else:# use the Regularized Least Squares Method 
            d = X.shape[1] # d number of features.
            I = np.identity(d+1) # d+1 accounting for X_tilda[0] (w_0).
            w = ( np.linalg.inv((X.T @ X) + (np.dot(lmbda, I))) ) @ (X.T @ y)
        # w = (x_t @ X + l.I)^-1 @ X_t @ Y
        # Return w; trained model parameters.
        return w