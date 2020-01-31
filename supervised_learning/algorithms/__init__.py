from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve, learning_curve
import numpy as np
import matplotlib.pyplot as plt
import time

'''
['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 
'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 
'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'max_error', 
'mutual_info_score', 'neg_brier_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 
'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 'normalized_mutual_info_score', 
'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 
'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'v_measure_score']
'''

class AbstractLearner():
    """A really useful base class for wrapping and abstracing the implementations of classifiers
    from the interface for calling them."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def classifier(self):
        """Defined by each learner subclass to return it's instance of a classifier."""
        pass

    def train(self, x, y):
        start = time.time()
        result = self.classifier().fit(x, y)
        end = time.time()
        print('\tfitting data took: {:.2f} secs'.format(end - start))
        return result

    def test(self, x):
        return self.classifier().predict(x)

    def score(self, x, y):
        return self.classifier().score(x, y)

    def probability(self, x):
        return self.classifier().predict_proba(x)

    def get_params(self):
        return self.classifier().get_params()

    def set_params(self, **params):
        return self.classifier().set_params(**params)

    def experiment(self, xtrain, xtest, ytrain, ytest):
        """Use a custom experiment defined by the learner."""
        print('No experiment defined!')

    @abstractmethod
    def tune(self):
        """Defined by the learners. Learners must return the params."""
        pass

    def _tune(self, params, x, y):
        """Search the parameter space for the classifier for the best tuned hyperparameters."""
        # https://www.pyimagesearch.com/2016/08/15/how-to-tune-hyperparameters-with-python-and-scikit-learn/
        grid = GridSearchCV(self.classifier(), params, verbose=1)
        grid.fit(x, y)
        return grid.best_params_

    @abstractmethod
    def get_validation_param(self):
        """Defined by the learners. Return a tuple of the validation parameter name and range."""
        pass

    def plot_learning_curve(self, x, y, scoring='accuracy'):
        """Plot the learning curve, a function of accuracy over N, the size of the data set."""
        # Important note on overfitting!
        # https://stats.stackexchange.com/questions/283738/sklearn-learning-curve-example
        print('Computing learning curve...')
        # https://scikit-learn.org/stable/modules/model_evaluation.html
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
            self.classifier(),
            x,
            y,
            scoring=scoring,
            cv=5,  # number of folds in cross-validation / number of points on the plots
            n_jobs=-1,  # number of cores to use
            return_times=True)
        # get the mean and std to "center" the plots
        train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
        test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
        fit_times_mean, fit_times_std = np.mean(fit_times, axis=1), np.std(fit_times, axis=1)

        plt.plot(train_sizes, train_mean, color="darkorange",  label="Training score")
        plt.plot(train_sizes, test_mean, color="navy", label="Cross-validation score")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")
        plt.title("Learning Curve: " + type(self).__name__)
        plt.xlabel("Training Set Size"), plt.ylabel(scoring), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


    def plot_validation_curve(self, x, y, param_name=None, param_range=None):
        """Plot the validation curve.
        The validation curve plots the influence of a single hyperparameter
        on the training score and test score to determine if the model is over or underfitting."""
        # param_name, param_range = self.get_validation_param()
        print('Computing validation curve...')
        train_scores, test_scores = validation_curve(
            self.classifier(),
            x,
            y,
            param_name=param_name,
            param_range=param_range,
            scoring="accuracy",
            n_jobs=-1)
        train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
        test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

        plt.title("Validation Curve: " + type(self).__name__)
        plt.plot(param_range, train_mean, label="Training score", color="darkorange")
        plt.plot(param_range, test_mean, label="Cross-validation score", color="navy")
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
        plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")
        plt.xlabel('Parameter: ' + param_name), plt.ylabel("Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
