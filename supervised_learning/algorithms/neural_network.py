from sklearn import neural_network

class NeuralNetworkLearner():
    """test"""
    def __init__(self):
        self.classifier = neural_network.MLPClassifier()

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
        pass
