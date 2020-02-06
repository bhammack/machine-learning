from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from . import AbstractLearner

class KNNLearner(AbstractLearner):
    def __init__(self):
        self.knn_classifier = KNeighborsClassifier(n_neighbors=10)

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

    def experiment(self, xtrain, xtest, ytrain, ytest):
        pass
        # self.plot_learning_curve(xtrain, ytrain)
        # self.plot_validation_curve(xtrain, ytrain, 'n_neighbors', np.arange(1, 51), 'roc_auc')
    
