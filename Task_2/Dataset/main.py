import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from lesson_2.gradient_descent import GradientDescent

columns = ['user_id', 'film_rating', 'film_id']

lines_number = 20352
data = pd.read_csv('data.csv', header=None, names=columns, nrows=lines_number)
encoder = OneHotEncoder(categories="auto")

# Применяем one-hot encoding
user_matrix = encoder.fit_transform(np.asarray(data['user_id']).reshape(-1,1))
film_matrix = encoder.fit_transform(np.asarray(data['film_id']).reshape(-1,1))

# Добавляем вектор единиц и формируем матрицу
ones = np.ones(shape=(lines_number, 1))
X = hstack([ones, user_matrix, film_matrix]).tocsr()
y = np.asarray(data['film_rating']).reshape(-1,1)

X,y = shuffle(X,y)

print(X.shape)
print(y.shape)

number_of_splits = 5
number_of_epochs = 100
factors_num = 2

errors = [0 for _ in range(number_of_splits)]
weights = [0 for _ in range(number_of_splits)]
factors = [0 for _ in range(number_of_splits)]

rmse_train = [0 for _ in range(number_of_splits)]
rmse_test = [0 for _ in range(number_of_splits)]

r2s_train = [0 for _ in range(number_of_splits)]
r2s_test = [0 for _ in range(number_of_splits)]

kf = KFold(n_splits=number_of_splits, shuffle=True)
kf.get_n_splits(X)

g_d = GradientDescent()

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Iteration {i}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    w_init = np.zeros((X.shape[1], 1))
    V_init = np.zeros((X.shape[1], factors_num))

    results = g_d.gradient_descent(X_train, y_train, w_init, V_init, eta=0.05, max_iter=number_of_epochs)
    weights[i], factors[i], errors[i] = results

    train_preds = g_d.make_prediction(X_train, weights[i], factors[i])
    test_preds = g_d.make_prediction(X_test, weights[i], factors[i])
# Performing Gradient Descent
# for i in range(epochs):
#     Y_pred = m*X + c  # The current predicted value of Y
#     D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
#     D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
#     m = m - L * D_m  # Update m
#     c = c - L * D_c  # Update c
