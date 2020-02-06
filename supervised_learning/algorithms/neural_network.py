from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace
from . import AbstractLearner

class NeuralNetworkLearner(AbstractLearner):
    """test"""
    def __init__(self):
        # https://analyticsindiamag.com/a-beginners-guide-to-scikit-learns-mlpclassifier/
        self.nn_classifier = MLPClassifier(n_iter_no_change=50, learning_rate='constant', hidden_layer_sizes=(100, 100), early_stopping=True, activation='logistic')
        # neural networks are iterative
        # digits:
        # {'activation': 'relu', 'hidden_layer_sizes': (100, 100)}
        # {'activation': 'tanh', 'hidden_layer_sizes': (50, 50)}
        # {'activation': 'relu', 'hidden_layer_sizes': (100, 100)}
        # adult:
        # {'activation': 'logistic', 'hidden_layer_sizes': 100}
        # {'activation': 'tanh', 'hidden_layer_sizes': (50, 50, 50)}
        # {'activation': 'logistic', 'hidden_layer_sizes': (100, 100)}

    def classifier(self):
        return self.nn_classifier

    def tune(self, x, y):
        # The most important aspect of a NN is it's depth and number of nodes.
        # https://datascience.stackexchange.com/questions/19768/how-to-implement-pythons-mlpclassifier-with-gridsearchcv
        params = {
            "activation": ['tanh', 'logistic', 'relu'],
            "hidden_layer_sizes": [(5), (10), (20), (35), (50), (75), (100), (150), (200), (50, 50), (50, 100)]
        }
        return self._tune(params, x, y)

    def plot_iterations_score(self):
        plt.title('MLPClassifier accuracy per iteration')
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.plot(self.nn_classifier.validation_scores_, color="darkorange", label="Training score")
        plt.legend(loc="best"), plt.tight_layout()
        plt.show()

    def experiment(self, xtrain, xtest, ytrain, ytest):
        # max_iter is not the iterations of the model / learning rate, it is the upper bound.
        # https://stackoverflow.com/questions/54024816/how-to-plot-learning-rate-vs-accuracy-sklearn
        # self.plot_learning_curve(xtrain, ytrain)
        # plt.plot(self.nn_classifier.loss_curve_)
        # self.plot_iterations_score()
        pass
        # self.plot_validation_curve(xtrain, ytrain, 'alpha', [0.00001, 0.0001, 0.001, 0.01, 0.1])
        # self.plot_validation_curve(xtrain, ytrain, 'learning_rate', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
        # self.plot_validation_curve(xtrain, ytrain, 'max_iter', np.arange(100, 500))
