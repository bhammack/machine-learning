from sklearn import svm
from . import AbstractLearner
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, learning_curve
from pdb import set_trace

class SVMLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.svm_classifier = svm.SVC()
        n = 4
        self.c_space = [10 ** x for x in range(-1 * n, n + 1)]
        self.kernels = ['rbf', 'linear', 'sigmoid', 'poly']
        self.kernel_colors = {
            'rbf': ('red', 'darkred'),
            'linear': ('blue', 'darkblue'),
            'sigmoid': ('green', 'darkgreen'),
            'poly': ('orange', 'darkorange')
        }

    def classifier(self):
        return self.svm_classifier

    # Data sets need to be scaled for SVM's
    # https://stats.stackexchange.com/questions/154224/when-using-svms-why-do-i-need-to-scale-the-features

    # What does C really mean in an SVM?
    # https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel

    def tune(self, x, y):
        params = {
            "C": self.c_space,
            "kernel": self.kernels
        }
        print('C space:', self.c_space)
        return self._tune(params, x, y)

    def plot_kernel_learning(self, xtrain, xtest, ytrain, ytest):
        """Plot the learning curve, a function of accuracy over N, the size of the data set."""
        # Important note on overfitting!
        # https://stats.stackexchange.com/questions/283738/sklearn-learning-curve-example
        print('Computing learning curve...')
        # https://scikit-learn.org/stable/modules/model_evaluation.html
        scoring = 'accuracy'
        for kernel in self.kernels:
            train_sizes, train_scores, test_scores = learning_curve(
                svm.SVC(kernel=kernel),
                xtrain,
                ytrain,
                scoring=scoring,
                cv=5,  # number of folds in cross-validation / number of points on the plots
                n_jobs=-1)
            # get the mean and std to "center" the plots
            train_mean = np.mean(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)

            plt.plot(train_sizes, train_mean, color=self.kernel_colors[kernel][0],  label="{} - Training score".format(kernel))
            plt.plot(train_sizes, test_mean, '--', color=self.kernel_colors[kernel][1], label="{} - Cross-validation score".format(kernel))


        plt.title("Learning Curves per kernel")
        plt.xlabel("Training Set Size"), plt.ylabel(scoring), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


    def plot_c_validation(self, xtrain, ytrain):
        """Plot the learning curve, a function of accuracy over N, the size of the data set."""
        # Important note on overfitting!
        # https://stats.stackexchange.com/questions/283738/sklearn-learning-curve-example
        # https://scikit-learn.org/stable/modules/model_evaluation.html
        scoring = 'accuracy'
        param_name = 'C'
        param_range = self.c_space
        for kernel in self.kernels:
            train_scores, test_scores = validation_curve(
                svm.SVC(kernel=kernel),
                xtrain,
                ytrain,
                param_name=param_name,
                param_range=param_range,
                scoring=scoring,
                n_jobs=-1)

            train_mean = np.mean(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)

            plt.plot(param_range, train_mean, color=self.kernel_colors[kernel][0],  label="{} - Training score".format(kernel))
            plt.plot(param_range, test_mean, '--', color=self.kernel_colors[kernel][1], label="{} - Cross-validation score".format(kernel))


        plt.title("Learning Curves per kernel")
        plt.xlabel("Training Set Size"), plt.ylabel(scoring), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def experiment(self, xtrain, xtest, ytrain, ytest):
        # self.plot_learning_curve(xtrain, ytrain)
        # self.plot_kernel_learning(xtrain, xtest, ytrain, ytest)
        self.plot_c_validation(xtrain, ytrain)