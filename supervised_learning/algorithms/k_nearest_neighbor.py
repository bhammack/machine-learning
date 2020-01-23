from sklearn.neighbors import KNeighborsClassifier
from . import AbstractLearner

class KNNLearner(AbstractLearner):
    def __init__(self):
        self.knn_classifier = KNeighborsClassifier()

    def classifier(self):
        return self.knn_classifier
