import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Training set
df = pd.read_csv(os.path.join(__location__, 'adult.data'), header=None)

indexes_to_encode = [1, 3, 5, 6, 7, 8, 9, 13, 14] # convert labels to indexes.
labels = {} # label index mapping for future reference
for col in indexes_to_encode:
    le = LabelEncoder()
    le.fit(df[col])
    labels[col] = le.classes_
    df[col] = le.transform(df[col])

# print(labels)
y = df.pop(14)
x = df

test_size = 0.33 # test data is 1/3 of original data set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
