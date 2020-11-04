import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
import pandas as pd
import numpy as np
import scipy.sparse as scp

from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, diags


class GradientDescent:

    def __init__(self,
                 learning_rate=1e-4, epochs=1e4, min_weight_dist=1e-4, factors_num=2,
                 weight=None, factor=None, c=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_weight_dist = min_weight_dist
        self.factors_num=factors_num
        self.weight = weight
        self.factor = factor
        self.c = c


    def predict(self, X, w=None, c=None):
        if w is None:
            w = self.weight
        if c is None:
            c = self.c

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        return np.dot(X, w) + c

    def make_prediction(self, X, w, V):
        a = np.sum(np.square(np.dot(X, V)), axis=1).reshape(-1, 1)
        b = np.sum(np.dot(X.power(2), np.square(V)), axis=1).reshape(-1, 1)

        return np.dot(X, w) + 0.5 * (a - b)

    def precompute_sum(self, X, V):
        return X.dot(V)

    def gradient_descent_step(self, X, y, w, V, eta=0.01):
        # Пересчет w
        w += (2 * eta / len(y)) * X.T.dot(y - X.dot(w))

        precomputed = self.precompute_sum(X, V)
        y_pred = self.make_prediction(X, w, V)

        # Пересчет V_i
        for i in range(V.shape[1]):
            a_diagonal = diags(np.array(precomputed)[:, i])
            a = a_diagonal.dot(X)

            b_diagonal = diags(V[:, i])
            b = X.power(2).dot(b_diagonal)

            V[:, i] += (2 * eta / X.shape[1]) * (a - b).T.dot(y - y_pred).reshape((-1,))

        return w, V

    def gradient_descent(self, X, y, w_init, V_init, eta=1e-2, max_iter=1e4):
        weight_dist = np.inf
        w = w_init
        V = V_init
        w_next = w_init
        V_next = V_init
        print('w_init shape:', w_init.shape)
        print('V_init shape:', V_init.shape)

        errors = list()
        for i in range(max_iter):
            w_next += (2 * eta / len(y)) * X.T.dot(y - X.dot(w))

            precomputed = np.dot(X, V)
            y_pred = self.make_prediction(X, w_next, V)

            # Пересчет V_i
            for j in range(V_next.shape[1]):
                a_diagonal = diags(np.array(precomputed)[:, j])
                a = a_diagonal.dot(X)

                b_diagonal = diags(V_next[:, j])
                b = X.power(2).dot(b_diagonal)

                V_next[:, j] += (2 * eta / X.shape[1]) * (a - b).T.dot(y - y_pred).reshape((-1,))

            print('w_next shape:', w_next.shape)
            print('V_next shape:', V_next.shape)
            y_pred = self.make_prediction(X, w_next, V_next)

            errors.append(MSE(y, y_pred))
            if i % 10 == 0:
                print(f"\tEpoch: {i}, MSE: {errors[i]}")

            w = w_next
            V = V_next

        return w, V, errors

    def fit(self, X, y):
        weight_dist = np.inf
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self.weight = np.zeros(X.shape[1])
        self.factor = np.zeros(X.shape[1], self.factors_num)
        self.c = 0
        iter = 0
        while weight_dist > self.min_weight_dist and iter < self.epochs:
            y_pred = self.predict(X)

            D_weight = (2 * self.learning_rate / len(y)) * np.dot(X.T, y - y_pred)
            D_c = (2 * self.learning_rate / len(y)) * sum(y - y_pred)

            self.weight += D_weight
            self.c += D_c

            weight_dist = D_weight.sum() + D_c

            iter += 1

    @property
    def feature_importances_(self):
        return np.append([self.c], self.weight)