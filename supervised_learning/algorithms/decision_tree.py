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
        self.dt_classifier = tree.DecisionTreeClassifier(max_depth=5)

    def classifier(self): # pylint: disable=E0202
        return self.dt_classifier

    def prune(self, x, y):
        """Prune a trained tree provided a cost-complexity parameter alpha."""
        path = self.dt_classifier.cost_complexity_pruning_path(x, y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        # Select one of the alphas according to the number of impurities in the nodes it relates to.
        print(ccp_alphas, impurities)
        alpha = ccp_alphas[0]
        # Re-fit the decision tree to the data set using the cost complexity alpha
        self.dt_classifier = tree.DecisionTreeClassifier(ccp_alpha=alpha)
        # self.train(x, y)  # retrain the model outside this class


    # Define the parameter space to search with GridSearchCV
    def tune(self, x, y):
        params = {
            "max_depth": np.arange(1, 51),
            "max_leaf_nodes": np.arange(2, 100) # max leaf nodes must be greater than 1
        }
        return self._tune(params, x, y)

    def export(self):
        """Exports the decision tree in a string tree structure."""
        return tree.export_text(self.dt_classifier.tree_)

    def get_validation_param(self):
        return ('max_leaf_nodes', np.arange(2, 101))
        # return ('max_depth', np.arange(1, 51))