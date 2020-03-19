import pandas as pd
import os
import numpy as np
from pdb import set_trace
from data import setup, encode, onehotencode, split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=[0, 1])

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Data set, now combined into one file
df = pd.read_csv(os.path.join(__location__, 'adult.data'), header=None)

# we have no header so our columns are indexes
cols_to_encode = [1, 3, 5, 6, 7, 8, 9, 13] # & 14 as label encode
cols_to_scale = [0, 2, 4, 10, 11, 12]
df = pd.get_dummies(df, columns=cols_to_encode)
df, labels = encode(df, [14])

# Scale the continuious values to between 0 and 1
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Pop off the class labels
y = df.pop(14).to_numpy()
x = df.to_numpy()


x_train, x_test, y_train, y_test = split(x, y, 0.33)