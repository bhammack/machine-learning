from sklearn import svm
from . import AbstractLearner

class SVMLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.svm_classifier = svm.SVC()

    def classifier(self):
        return self.svm_classifier

    def tune(self, x, y):
        params = {
            "kernel": ['rbf', 'sigmoid'],
            "gamma": [] # TODO this
        }
        return self._tune(params, x, y)
