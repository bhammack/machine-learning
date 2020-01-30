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


def get_data_set():
    if args.adult:
        return adult.x_train, adult.x_test, adult.y_train, adult.y_test, adult.x, adult.y
    elif args.digits:
        return digits.x_train, digits.x_test, digits.y_train, digits.y_test, digits.x, digits.y


def experiment(learner):
    """Run the experiment with the specified learner and data set."""
    start = time.time()
    xtrain, xtest, ytrain, ytest, x, y = get_data_set()
    print('Learner is using a', type(learner.classifier()).__name__, 'classifier with params...')
    print(learner.get_params())
    learner.train(xtrain, ytrain)
    # print('Error:\t', learner.classifier().score(xtrain, ytrain))
    result = learner.test(xtest)
    # probs = learner.probability(xtest)
    # print('Probability:\t', probs)

    # Print the score of the trained classifier on the cross-validation set.
    # For most models, this score should be the exact same. If they differ, check the classifier docs.
    # https://stackoverflow.com/questions/40726899/difference-between-score-vs-accuracy-score-in-sklearn
    print('=' * 15 + '[ SCORE ]' + '=' * 15)
    print('Classifier Score: ', learner.score(xtest, ytest))
    print('Accuracy Score:   ', accuracy_score(result, ytest))

    if args.search:
        print("Tuning model to search space. Hold on to your shorts!")
        result = learner.tune(xtrain, ytrain)
        print(result)

    learner.experiment(xtrain, xtest, ytrain, ytest)
    # learner.plot_learning_curve(xtrain, ytrain)
    # learner.plot_validation_curve(xtrain, ytrain)

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
