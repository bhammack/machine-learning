import argparse
from data import adult
from supervised_learning.algorithms import decision_tree
from sklearn.metrics import accuracy_score

def main():
    print('Hello world!')
    dt = decision_tree.DecisionTreeLearner()
    dt.train(adult.x_train, adult.y_train)
    result = dt.test(adult.x_test)
    score = accuracy_score(result, adult.y_test)
    print(score)
    print(dt.export())


if __name__ == '__main__':
    main()
