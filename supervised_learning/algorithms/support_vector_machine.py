from sklearn import svm
from sklearn.linear_model import stochastic_gradient, SGDClassifier
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np
from . import AbstractLearner

class SVMLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.svm_classifier = svm.SVC()

    def classifier(self):
        return self.svm_classifier
