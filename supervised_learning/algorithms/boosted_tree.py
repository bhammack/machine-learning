from sklearn.ensemble import AdaBoostClassifier
# Adaboost requires a tree, so I need to feed it the decisiontree from the other learner
from . import AbstractLearner

class BoostedTreeLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.bdt_classifier = AdaBoostClassifier()
        # boosted trees are iterative

    def classifier(self):
        return self.bdt_classifier

    def experiment(self, xtrain, xtest, ytrain, ytest):
        # plot accuracy vs weak learner count
        