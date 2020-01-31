from sklearn.neural_network import MLPClassifier
import numpy as np
from . import AbstractLearner

class NeuralNetworkLearner(AbstractLearner):
    """test"""
    def __init__(self):
        # https://analyticsindiamag.com/a-beginners-guide-to-scikit-learns-mlpclassifier/
        self.nn_classifier = MLPClassifier(learning_rate='adaptive')
        # neural networks are iterative

    def classifier(self):
        return self.nn_classifier

    def tune(self, x, y):
        # The most important aspect of a NN is it's depth and number of nodes.
        # https://datascience.stackexchange.com/questions/19768/how-to-implement-pythons-mlpclassifier-with-gridsearchcv
        params = {
            'learning_rate': ["constant", "invscaling", "adaptive"],
            "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
            # no one appears to search for max_iters...
            "hidden_layer_sizes": [(50,50,50),(50,100,50),(100,)] # TODO this
        }
        return self._tune(params, x, y)

    def experiment(self, xtrain, xtest, ytrain, ytest):
        # max_iter is not the iterations of the model / learning rate, it is the upper bound.
        # https://stackoverflow.com/questions/54024816/how-to-plot-learning-rate-vs-accuracy-sklearn
        self.plot_learning_curve(xtrain, ytrain)
        # self.plot_validation_curve(xtrain, ytrain, 'alpha', [0.00001, 0.0001, 0.001, 0.01, 0.1])
        self.plot_validation_curve(xtrain, ytrain, 'learning_rate', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
        # self.plot_validation_curve(xtrain, ytrain, 'max_iter', np.arange(100, 500))
