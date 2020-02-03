from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# Adaboost requires a tree, so I need to feed it the decisiontree from the other learner
from . import AbstractLearner
import numpy as np

class BoostedTreeLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.bdt_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1.5)
        # boosted trees are iterative

    def classifier(self):
        return self.bdt_classifier

    def experiment(self, xtrain, xtest, ytrain, ytest):
        # plot accuracy vs weak learner count
        pass
        self.plot_validation_curve(xtrain, ytrain, 'n_estimators', np.arange(2, 151))
        # self.plot_validation_curve(xtrain, ytrain, 'learning_rate', np.arange(1.0, 11))
