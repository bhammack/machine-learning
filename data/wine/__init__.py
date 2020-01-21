import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# This is perhaps the best known database to be found in the pattern recognition literature.
# Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# Predicted attribute: class of iris plant.

# Data set
df = pd.read_csv(os.path.join(__location__, 'winequality-red.csv'), delimiter=';')

indexes_to_encode = [] # convert labels to indexes.
labels = {} # label index mapping for future reference
for col in indexes_to_encode:
    le = LabelEncoder()
    le.fit(df[col])
    labels[col] = le.classes_
    df[col] = le.transform(df[col])

y = df.pop('quality') # pop off the classes to predict
x = df # x is just the rest of the data frame

test_size = 0.33 # test data is 1/3 of original data set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

