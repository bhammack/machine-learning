import argparse
from data import adult
from data import iris
from data import wine
from supervised_learning.algorithms import decision_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import sys



def plot_validation_curve():
    plt.title('Validation Curve')
    plt.xlabel('idk x label')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.1)
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


def dt():
    """Run the decision tree experiment."""
    print('Running the decision tree experiment...')
    dt = decision_tree.DecisionTreeLearner()
    dt.train(adult.x_train, adult.y_train)
    result = dt.test(adult.x_test)
    score = accuracy_score(result, adult.y_test)
    print('Adult', score)

    dt = decision_tree.DecisionTreeLearner()
    dt.train(iris.x_train, iris.y_train)
    result = dt.test(iris.x_test)
    score = accuracy_score(result, iris.y_test)
    print('Iris', score)

    dt = decision_tree.DecisionTreeLearner()
    dt.train(wine.x_train, wine.y_train)
    result = dt.test(wine.x_test)
    score = accuracy_score(result, wine.y_test)
    print('Wine', score)


def knn():
    """Run the knn experiment."""
    print('Running the k-nearest neighbors experiment...')


def svm():
    """Run the svm experiment."""
    print('Running the support vector machine experiment...')


def ann():
    """Run the neural network experiment."""
    print('Running the neural network experiment...')


def bdt():
    """Run the boosting experiment."""
    print('Running the boosted decision tree experiment...')


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser(description='Select an experiment to run')
    parser.add_argument('--dt', action='store_true', help='Run the decision tree classifier experiment')
    parser.add_argument('--knn', action='store_true', help='Run the k-nearest neighbors experiment')
    parser.add_argument('--ann', action='store_true', help='Run the artificial neural network experiment')
    parser.add_argument('--svm', action='store_true', help='Run the support vector machine experiment')
    parser.add_argument('--bdt', action='store_true', help='Run the boosted decision tree classifier experiment')
    args = parser.parse_args()

    if args.dt:
        dt()
    if args.knn:
        knn()
    if args.ann:
        ann()
    if args.svm:
        svm()
    if args.bdt:
        bdt()
    if len(sys.argv) == 1:
        parser.print_help()
    end = time.time()

    print('Experiment duration:', str(end - start))