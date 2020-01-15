import argparse
from data import adult
from supervised_learning.algorithms import decision_tree
from sklearn.metrics import accuracy_score


def dt():
    """Run the decision tree experiment."""
    print('Hello world!')
    dt = decision_tree.DecisionTreeLearner()
    dt.train(adult.x_train, adult.y_train)
    result = dt.test(adult.x_test)
    score = accuracy_score(result, adult.y_test)
    print(score)
    print(dt.export())


def knn():
    """Run the knn experiment."""
    pass


def svm():
    """Run the svm experiment."""
    pass


def nn():
    """Run the neural network experiment."""
    pass


def gbc():
    """Run the boosting experiment."""
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select an experiment to run')
    parser.add_argument('--dt', action='store_true', help='Run the Decision Tree experiment')
    parser.add_argument('--knn', action='store_true', help='Run the KNN experiment')
    parser.add_argument('--nn', action='store_true', help='Run the Neural Network experiment')
    parser.add_argument('--svm', action='store_true', help='Run the SVM experiment')
    parser.add_argument('--gbc', action='store_true', help='Run the Boosting experiment')
    args = parser.parse_args()

    if args.dt:
        dt()
    if args.knn:
        knn()
    if args.nn:
        nn()
    if args.svm:
        svm()
    if args.gbc:
        gbc()
