import sklearn.model_selection as ms
from sklearn import tree
from subprocess import call
from . import AbstractLearner

class DecisionTreeLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.dt_classifier = tree.DecisionTreeClassifier()

    def classifier(self): # pylint: disable=E0202
        return self.dt_classifier

    def prune(self):
        """Prune a trained tree."""
        pass

    def export(self):
        """Exports the decision tree in a string tree structure."""
        return tree.export_text(self.dt_classifier.tree_)
