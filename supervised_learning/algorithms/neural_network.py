from sklearn.neural_network import MLPClassifier
import numpy as np
from . import AbstractLearner

class NeuralNetworkLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.nn_classifier = MLPClassifier()

    def classifier(self):
        return self.nn_classifier

    def tune(self, x, y):
        params = {
            "alpha": np.arange(1, 25),
            "max_iter": np.arange(1, 200),
            # "hidden_layer_sizes": [] # TODO this
        }
        return self._tune(params, x, y)