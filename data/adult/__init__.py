import pandas as pd
import os
import numpy as np
from data import setup, encode

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Data set, now combined into one file
df = pd.read_csv(os.path.join(__location__, 'adult.data'), header=None)

# we have no header so our columns are indexes
cols_to_encode = [1, 3, 5, 6, 7, 8, 9, 13, 14]
df, labels = encode(df, cols_to_encode)

y = df.pop(14)
x = df

x_train, x_test, y_train, y_test = setup(x, y)
# This data set is imbalanced!
# print('y_train mean', np.mean(y_train))
# print('y_test mean', np.mean(y_test))
best_params_knn = {"n_neighbors": 19, "p": 1}
