
import clustering
import reduction
import sys
from data import adults, digits
import argparse
import numpy as np

def use_data_set():
    if args.adults:
        print('> analyzing the adults data set...')
        return adults.x_train, adults.x_test, adults.y_train, adults.y_test, adults.x, adults.y
    elif args.digits:
        print('> analyzing the digits data set...')
        return digits.x_train, digits.x_test, digits.y_train, digits.y_test, digits.x, digits.y


def use_clustering_algo(Y):
    k = len(np.unique(Y))
    print('> found {} unique labels in the data...'.format(k))
    if args.kmeans:
        print('> using k-means clustering...')
        c = clustering.KMeansClustering(k)
    if args.em:
        print('> using expectation maximization clustering...')
        c = clustering.EMClustering(k)
    return c


def cluster():
    print('Clustering the data...')
    xtrain, xtest, ytrain, ytest, X, Y = use_data_set()
    clustering = use_clustering_algo(Y)
    clustering.experiment(xtrain, xtest, ytrain, ytest, X, Y)


def dim_reduce():
    print('Reducing the dimensions of the data...')


def cluster_and_reduce():
    print('Clustering and reducing the data...')


def cluster_and_nn():
    print('Clustering data and running the neural network...')


def reduce_and_nn():
    print('Applying dimension reduction and running the neural network...')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select an experiment to run')
    parser.add_argument('--kmeans', action='store_true', help='Use the k-means clustering algorithm')
    parser.add_argument('--em', action='store_true', help='Use the expectation maximization algorithm')

    parser.add_argument('--pca', action='store_true', help='Reduce dimensions using PCA')
    parser.add_argument('--ica', action='store_true', help='Reduce dimensions using ICA')
    parser.add_argument('--proj', action='store_true', help='Reduce dimensions using randomized projections')
    parser.add_argument('--svd', action='store_true', help='Reduce dimensions using single-value decomposition')

    parser.add_argument('--nn', action='store_true', help='Run the neural network on the resultant data')

    parser.add_argument('--adults', action='store_true', help='Experiment with the U.S. adults data set')
    parser.add_argument('--digits', action='store_true', help='Experiment with the handwritten digits data set')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
    do_clustering = args.kmeans or args.em
    do_reduction = args.pca or args.ica or args.proj or args.svd
    do_nn = args.nn

    if do_clustering and not do_reduction:
        # Only do clustering
        cluster()
    if do_reduction and not do_clustering:
        # Only do reduction
        dim_reduce()
    if do_clustering and do_reduction:
        # Do both clustering and reduction
        cluster_and_reduce()
    if do_reduction and do_nn and not do_clustering:
        # Run a nn on the reduced dimensions
        reduce_and_nn()
    if do_clustering and do_nn and not do_reduction:
        # Run a nn on the clusted data as if it were dim reduction
        cluster_and_nn()
