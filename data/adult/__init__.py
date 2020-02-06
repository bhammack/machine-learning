import pandas as pd
import os
import numpy as np
from pdb import set_trace
from data import setup, encode, onehotencode

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Data set, now combined into one file
df = pd.read_csv(os.path.join(__location__, 'adult.data'), header=None)

# we have no header so our columns are indexes
cols_to_encode = [1, 3, 5, 6, 7, 8, 9, 13] # & 14 as label encode
df = pd.get_dummies(df, columns=cols_to_encode)
df, labels = encode(df, [14])



# Code to force balance among target class
print('Balancing Adult data set... please wait')
greater_than_50k = np.where(df[14] == 1)[0] # 11K
less_than_50k = np.where(df[14] == 0)[0]
lt_samples = np.random.choice(less_than_50k, greater_than_50k.shape[0]) # match the less than 50k to greater than 50k in size
balanced = []
for index in greater_than_50k:
    balanced.append(df.iloc[index])
for index in lt_samples:
    balanced.append(df.iloc[index])
balanced = pd.DataFrame(data=balanced, columns=df.columns)
df = balanced



# Reduce the size of the data using a random sample for performance reasons
# df = df.sample(4096)


y = df.pop(14)
x = df


x_train, x_test, y_train, y_test = setup(x, y)
# This data set is imbalanced!
# print('y_train mean', np.mean(y_train))
# print('y_test mean', np.mean(y_test))
best_params_knn = {"n_neighbors": 19, "p": 1}
