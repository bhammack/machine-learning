from sklearn import svm
from . import AbstractLearner

class SVMLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.svm_classifier = svm.SVC()

    def classifier(self):
        return self.svm_classifier

    # Data sets need to be scaled for SVM's
    # https://stats.stackexchange.com/questions/154224/when-using-svms-why-do-i-need-to-scale-the-features

    def tune(self, x, y):
        params = {
            "c": [],
            "kernel": ['rbf', 'linear', 'sigmoid'],
        }
        return self._tune(params, x, y)
