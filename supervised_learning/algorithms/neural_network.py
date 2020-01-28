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
        # The most important aspect of a NN is it's depth and number of nodes.
        params = {
            "alpha": np.arange(1, 25), # what shoudl this be?
            "max_iter": np.arange(1, 200), # what should this be?
            "hidden_layer_sizes": [] # TODO this
        }
        return self._tune(params, x, y)

    def get_validation_param(self):
        return ('max_iter', np.arange(1, 51, 3))