from sklearn.datasets import load_digits
from data import setup

x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = setup(x, y)

best_params_dt = {'max_depth': 13, 'max_leaf_nodes': 93}
best_params_knn = {'metric': 'manhattan', 'n_neighbors': 3}
best_params_nn = {'alpha': 1e-05, 'hidden_layer_sizes': (100,), 'learning_rate': 'invscaling'}
