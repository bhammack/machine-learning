import argparse
from data import adult
from data import wine
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import sys
from supervised_learning.algorithms.decision_tree import DecisionTreeLearner
from supervised_learning.algorithms.k_nearest_neighbor import KNNLearner
from supervised_learning.algorithms.boosted_tree import BoostedTreeLearner
from supervised_learning.algorithms.neural_network import NeuralNetworkLearner
from supervised_learning.algorithms.support_vector_machine import SVMLearner

def plot_validation_curve():
    plt.title('Validation Curve')
    plt.xlabel('idk x label')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.1)
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def experiment(learner):
    start = time.time()

    if args.adult:
        learner.train(adult.x_train, adult.y_train)
        result = learner.test(adult.x_test)
        score = accuracy_score(result, adult.y_test)
        print('Adult', score)

    if args.wine:
        learner.train(wine.x_train, wine.y_train)
        result = learner.test(wine.x_test)
        score = accuracy_score(result, wine.y_test)
        print('Wine', score)

    end = time.time()
    print('Experiment duration: {:.2f} secs'.format(end - start))

def dt():
    """Run the decision tree experiment."""
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
    parser.add_argument('--wine', action='store_true', help='Experiment with the wine data set')


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
