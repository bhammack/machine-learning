from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV

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
        return self.classifier().fit(x, y)

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
