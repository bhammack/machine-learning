import argparse
from data import adult
from supervised_learning.algorithms import decision_tree


def main():
    print('Hello world!')
    # print(adult.x_train)
    # print(adult.y_train)
    dt = decision_tree.DecisionTreeLearner()
    # print(adult.df_train)
    # dt.train(adult.x_train, adult.y_train)

if __name__ == '__main__':
    main()
