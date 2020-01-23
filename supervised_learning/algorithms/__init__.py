from abc import ABC, abstractmethod

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

    def get_params(self):
        return self.classifier().get_params()

    def set_params(self, **params):
        return self.classifier().set_params(**params)
