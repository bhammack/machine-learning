import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Training set
df_train = pd.read_csv(os.path.join(__location__, 'adult.data'), header=None)
indexes_to_encode = [1, 3, 5, 6, 7, 8, 9, 13, 14] # convert labels to indexes.
train_labels = {} # label index mapping for future reference
for col in indexes_to_encode:
    le = LabelEncoder()
    le.fit(df_train[col])
    train_labels[col] = le.classes_
    df_train[col] = le.transform(df_train[col])
# print(df_train)


# Testing set
# Training set
df_test = pd.read_csv(os.path.join(__location__, 'adult.test'), header=None)
test_labels = {} # label index mapping for future reference
for col in indexes_to_encode:
    le = LabelEncoder()
    le.fit(df_test[col])
    test_labels[col] = le.classes_
    df_test[col] = le.transform(df_test[col])
# print(df_test)

# print(train_labels)
# print(test_labels)

# labels = label_encoder.fit_transform(df_train)
# print(labels)

y_train = df_train.pop(14)
x_train = df_train
y_test = df_test.pop(14)
x_test = df_test

