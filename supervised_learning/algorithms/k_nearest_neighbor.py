from sklearn.neighbors import KNeighborsClassifier
from . import AbstractLearner

class KNNLearner(AbstractLearner):
    def __init__(self):
        self.knn_classifier = KNeighborsClassifier()

    def classifier(self):
        return self.knn_classifier

    # For KNN, there are really two hyperparameters we need to tune
    # 1. the number of neighbors, K
    # 2. the distance/similarity function

    def testaaa(self, x, y):
        params = {"n_neighbors": np.arange(1, 31, 2),"metric": ["euclidean", "cityblock"]}
        grid = GridSearchCV(self.knn_classifier, params)
        grid.fit(x, y)
