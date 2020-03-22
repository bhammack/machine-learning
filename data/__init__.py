from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from pdb import set_trace

def encode(df, cols_to_encode):
    labels = {} # label index mapping for future reference
    for col in cols_to_encode:
        le = LabelEncoder()
        le.fit(df[col])
        labels[col] = le.classes_
        df[col] = le.transform(df[col])
    return df, labels

def onehotencode(df, cols_to_encode):
    set_trace()
    
    one = OneHotEncoder(categories=cols_to_encode)
    ct = ColumnTransformer()

    result = one.fit_transform(df)
    
    set_trace()
    return df, None


def split(x, y, test_size=0.33):
    return train_test_split(x, y, test_size=test_size)


def scale(x_train, x_test):
    # scale the data sets. SPLIT THEN SCALE!
    # Scaling then splitting results in data leakage
    # your testing set will be influenced/scaled by values in your training set
    scaleyboi = StandardScaler()
    scaleyboi.fit(x_train)
    x_train = scaleyboi.transform(x_train)
    x_test = scaleyboi.transform(x_test)
    return x_train, x_test


def setup(x, y, test_size=0.33):
    """Called by data set init. Given the raw data, split and scale it."""
    x_train, x_test, y_train, y_test = split(x, y, test_size)
    x_train, x_test = scale(x_train, x_test)
    return x_train, x_test, y_train, y_test
