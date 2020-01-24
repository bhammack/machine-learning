from sklearn import svm
from . import AbstractLearner

class SVMLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.svm_classifier = svm.SVC()

    def classifier(self):
        return self.svm_classifier
