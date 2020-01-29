from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from . import AbstractLearner

class KNNLearner(AbstractLearner):
    def __init__(self):
        self.knn_classifier = KNeighborsClassifier()

    def classifier(self):
        return self.knn_classifier

    # For KNN, there are really two hyperparameters we need to tune
    # 1. the number of neighbors, K
    # 2. the distance/similarity function

    def tune(self, x, y):
        params = {
            "n_neighbors": np.arange(1, 51),
            "metric": ["manhattan", "euclidean", "chebyshev"]
        }
        return self._tune(params, x, y)


    def get_validation_param(self):
        return ('n_neighbors', np.arange(1, 51))
