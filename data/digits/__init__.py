from sklearn.datasets import load_digits
from data import setup

x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = setup(x, y)
