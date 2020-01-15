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

    def export(self):
        """Exports the decision tree in a string tree structure."""
        return tree.export_text(self.classifier.tree_)
