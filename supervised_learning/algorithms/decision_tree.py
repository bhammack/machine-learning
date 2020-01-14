import numpy as np
import sklearn.model_selection as ms
from sklearn import tree
from subprocess import call

class DecisionTreeLearner():
    """test"""
    def __init__(self):
        self.classifier = tree.DecisionTreeClassifier()

    def train(self, x, y):
        """Train the learner on the training data set."""
        self.classifier = self.classifier.fit(x, y)
        return self

    def prune(self):
        """Prune a trained tree."""

    def test(self, data):
        """Test the learned tree on a test data set."""
        return self.classifier.predict(data)

    def export(self, filename):
        """Exports the decision tree as a graphviz DOT file."""
        tree.export_graphviz(self.classifier.tree_, out_file=filename)
        call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])
