import numpy as np
import sklearn.model_selection as ms
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeLearner():
    """test"""
    def __init__(self):
        self.classifier = DecisionTreeClassifier()

    def train(self, x, y):
        """Train the learner on the training data set."""
        self.classifier = self.classifier.fit(x, y)
        return self

    def prune(self):
        """Prune a trained tree."""

    def test(self, data):
        """Test the learned tree on a test data set."""
        return self.classifier.predict(data)







