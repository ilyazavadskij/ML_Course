from sklearn.metrics import mean_squared_error as MSE
import numpy as np

import numpy as np
from scipy.sparse import diags
from sklearn.metrics import mean_squared_error as MSE


class GradientDescent:

    def __init__(self,
                 learning_rate=1e-4, epochs=1e4, min_weight_dist=1e-4, factors_num=2,
                 weight=None, factor=None, error=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_weight_dist = min_weight_dist
        self.factors_num = factors_num
        self.weight = weight
        self.factor = factor
        self.error = error


    def predict(self, X, w=None, factor=None):
        if w is None:
            w = self.weight
        if factor is None:
            factor = self.factor

        return X.dot(w) + (np.sum(np.square(X.dot(factor)), axis=1).reshape(-1, 1) - np.sum(X.power(2).dot(np.square(factor)), axis=1).reshape(-1, 1)) / 2


    def fit(self, X, y):
        self.weight = np.zeros((X.shape[1], 1))
        self.factor = np.zeros((X.shape[1], self.factors_num))

        w_next = self.weight
        f_next = self.factor
        for i in range(int(self.epochs)):
            w_next += (2 * self.learning_rate / len(y)) * X.T.dot(y - X.dot(self.weight))

            precomputation = X.dot(self.factor)
            self.weight = w_next
            y_pred = self.predict(X)
            for j in range(f_next.shape[1]):
                a = diags(np.array(precomputation)[:, j]).dot(X)
                b = X.power(2).dot(diags(f_next[:, j]))
                f_next[:, j] += (2 * self.learning_rate / X.shape[1]) * (a - b).T.dot(y - y_pred).reshape((-1,))

            self.factor = f_next

            y_pred = self.predict(X)

            # errors.append(MSE(y, y_pred))
            # if i % 10 == 0:
            #     print(f"\tEpoch: {i}, MSE: {errors[i]}")

        return self.weight, self.factor, self.error
