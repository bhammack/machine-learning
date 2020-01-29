from sklearn import svm
from . import AbstractLearner

class SVMLearner(AbstractLearner):
    """test"""
    def __init__(self):
        self.svm_classifier = svm.SVC()
        n = 3
        self.c_space = [10 ** x for x in range(-1 * n, n + 1)]
        self.gamma_space = [10 ** x for x in range(-4, 4)]

    def classifier(self):
        return self.svm_classifier

    # Data sets need to be scaled for SVM's
    # https://stats.stackexchange.com/questions/154224/when-using-svms-why-do-i-need-to-scale-the-features

    # What does C really mean in an SVM?
    # https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel

    def tune(self, x, y):
        params = {
            "C": self.c_space,
            "kernel": ['rbf', 'linear', 'sigmoid'],
            "gamma": self.gamma_space
        }
        return self._tune(params, x, y)

    def get_validation_param(self):
        # Scans the exponential magnitude space of C values...
        return ('gamma', self.gamma_space)
        # return ('max_depth', np.arange(1, 51))