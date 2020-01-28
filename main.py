import argparse
from data import adult, wine, digits
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import sys
from pdb import set_trace as debug
from supervised_learning.algorithms.decision_tree import DecisionTreeLearner
from supervised_learning.algorithms.k_nearest_neighbor import KNNLearner
from supervised_learning.algorithms.boosted_tree import BoostedTreeLearner
from supervised_learning.algorithms.neural_network import NeuralNetworkLearner
from supervised_learning.algorithms.support_vector_machine import SVMLearner
import numpy as np

from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import plot_roc_curve


def plot_validation_curve(learner, x, y):
    """Plot the validation curve."""
    param_name, param_range = learner.get_validation_param()
    print('Computing validation curve...')
    train_scores, test_scores = validation_curve(
        learner.classifier(),
        x,
        y,
        param_name=param_name,
        param_range=param_range,
        scoring="accuracy",
        n_jobs=-1)
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

    plt.title("Validation Curve: " + type(learner).__name__)
    plt.plot(param_range, train_mean, label="Training score", color="darkorange")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="navy")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")
    plt.xlabel('Parameter: ' + param_name), plt.ylabel("Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(learner, x, y):
    """Plot the learning curve, a function of accuracy over N, the size of the data set."""
    print('Computing learning curve...')
    train_sizes, train_scores, test_scores = learning_curve(
        learner.classifier(),
        x,
        y,
        cv=5,  # number of folds in cross-validation
        n_jobs=-1,  # number of cores to use
        scoring="accuracy")
    # get the mean and std to "center" the plots
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color="darkorange",  label="Training score")
    plt.plot(train_sizes, test_mean, color="navy", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")
    plt.title("Learning Curve: " + type(learner).__name__)
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def get_data_set():
    if args.adult:
        return adult.x_train, adult.x_test, adult.y_train, adult.y_test
    elif args.digits:
        return digits.x_train, digits.x_test, digits.y_train, digits.y_test


def experiment(learner):
    """Run the experiment with the specified learner and data set."""
    start = time.time()
    xtrain, xtest, ytrain, ytest = get_data_set()
    print('Learner is using a', type(learner.classifier()).__name__, 'classifier with params...')
    print(learner.get_params())
    learner.train(xtrain, ytrain)
    result = learner.test(xtest)
    score = accuracy_score(result, ytest)
    print('Score:\t', score)

    if args.search:
        print("Tuning model to search space. Hold on to your shorts!")
        result = learner.tune(xtrain, ytrain)
        print(result)

    plot_learning_curve(learner, xtrain, ytrain)
    plot_validation_curve(learner, xtrain, ytrain)
    # debug()
    end = time.time()
    print('Experiment duration: {:.2f} secs'.format(end - start))

def dt():
    """Run the decision tree experiment."""
    print('Running the decision tree experiment...')
    experiment(DecisionTreeLearner())


def knn():
    """Run the knn experiment."""
    print('Running the k-nearest neighbors experiment...')
    experiment(KNNLearner())


def svm():
    """Run the svm experiment."""
    print('Running the support vector machine experiment...')
    experiment(SVMLearner())


def nn():
    """Run the neural network experiment."""
    print('Running the neural network experiment...')
    experiment(NeuralNetworkLearner())


def bdt():
    """Run the boosting experiment."""
    print('Running the boosted decision tree experiment...')
    experiment(BoostedTreeLearner())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Select an experiment to run')
    parser.add_argument('--dt', action='store_true', help='Run the decision tree classifier experiment')
    parser.add_argument('--knn', action='store_true', help='Run the k-nearest neighbors experiment')
    parser.add_argument('--nn', action='store_true', help='Run the artificial neural network experiment')
    parser.add_argument('--svm', action='store_true', help='Run the support vector machine experiment')
    parser.add_argument('--bdt', action='store_true', help='Run the boosted decision tree classifier experiment')

    parser.add_argument('--adult', action='store_true', help='Experiment with the adult data set')
    parser.add_argument('--digits', action='store_true', help='Experiment with the handwritten digits data set')

    parser.add_argument('--search', action='store_true', help='Search for the best parameter set')



    args = parser.parse_args()

    if args.dt:
        dt()
    if args.knn:
        knn()
    if args.nn:
        nn()
    if args.svm:
        svm()
    if args.bdt:
        bdt()
    if len(sys.argv) == 1:
        parser.print_help()
