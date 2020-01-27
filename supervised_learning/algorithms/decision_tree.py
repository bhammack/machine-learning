import sklearn.model_selection as ms
from sklearn import tree
from subprocess import call
import numpy as np
from . import AbstractLearner

# Pruning
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

class DecisionTreeLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.dt_classifier = tree.DecisionTreeClassifier()

    def classifier(self): # pylint: disable=E0202
        return self.dt_classifier

    def prune(self, x, y):
        """Prune a trained tree provided a cost-complexity parameter alpha."""
        path = self.dt_classifier.cost_complexity_pruning_path(x, y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        # Select one of the alphas according to the number of impurities in the nodes it relates to.
        alpha = ccp_alphas[0]
        # Re-fit the decision tree to the data set using the cost complexity alpha
        self.dt_classifier = tree.DecisionTreeClassifier(ccp_alpha=alpha)
        # self.train(x, y)  # retrain the model outside this class


    # TODO: this
    def tune(self, x, y):
        params = {
            "min_samples_leaf": np.arange(1, 25, 1),
            "max_depth": [1, 2] # when p = 1, use manhattan distance. p = 2 is euclidean.
        }
        return self._tune(params, x, y)

    def export(self):
        """Exports the decision tree in a string tree structure."""
        return tree.export_text(self.dt_classifier.tree_)

    def get_validation_param(self):
        return ('min_samples_leaf', np.arange(1, 25, 1))