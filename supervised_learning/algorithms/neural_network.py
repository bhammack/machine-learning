from sklearn.neural_network import MLPClassifier
from . import AbstractLearner

class NeuralNetworkLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.nn_classifier = MLPClassifier()

    def classifier(self):
        return self.nn_classifier
