import sklearn.model_selection as ms
from sklearn import tree
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt

from . import AbstractLearner

# Pruning
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

class DecisionTreeLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.dt_classifier = tree.DecisionTreeClassifier()
        # self.dt_classifier = tree.DecisionTreeClassifier(max_depth=5)

    def classifier(self): # pylint: disable=E0202
        return self.dt_classifier

    def prune(self, x, y):
        """Prune a trained tree provided a cost-complexity parameter alpha."""
        # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

        # STATS BEFORE PRUNING

        path = self.dt_classifier.cost_complexity_pruning_path(x, y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        # Select one of the alphas according to the number of impurities in the nodes it relates to.
        # alphas are in ascending order, so let's get the biggest one.
        # we can't take the biggest alpha because that is the trivial tree of only one node.
        # Select the alpha that will remove roughly half of the nodes in the tree.
        alpha = ccp_alphas[len(ccp_alphas) // 2]
        # Re-fit the decision tree to the data set using the cost complexity alpha
        self.dt_classifier = tree.DecisionTreeClassifier(ccp_alpha=alpha)
        # self.train(x, y)  # retrain the model outside this class

        plt.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
        plt.xlabel("effective alpha")
        plt.ylabel("total impurity of leaves")
        plt.title("Total Impurity vs effective alpha for training set")
        plt.tight_layout()
        plt.show()


    def experiment(self, xtrain, xtest, ytrain, ytest):
        print('Score pre-pruning tree:', self.dt_classifier.score(xtest, ytest))
        print('Node count pre-pruning:', self.dt_classifier.tree_.node_count)
        self.prune(xtrain, ytrain)
        self.train(xtrain, ytrain)
        print('Score post-pruning tree:', self.dt_classifier.score(xtest, ytest))
        print('Node count post-pruning:', self.dt_classifier.tree_.node_count)



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