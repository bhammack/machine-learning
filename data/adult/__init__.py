import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Training set
df_train = pd.read_csv(os.path.join(__location__, 'adult.data'), header=None)
# indexes_to_encode = [1, 3, 14]
for col in indexes_to_encode:
    print(col)

# label_encoder = preprocessing.LabelEncoder()
# labeled_df = df_train[indexes_to_encode]
# labels = label_encoder.fit_transform(df_train)
# print(labels)

# y_train = df_train.pop(14)
# x_train = df_train

