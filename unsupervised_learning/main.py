
import clustering
import reduction
import sys
from data import adults, digits
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
#
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import metrics
#
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.decomposition import TruncatedSVD

# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py

data_set = None

def use_data_set():
    global data_set
    if args.adults:
        print('> analyzing the adults data set...')
        data_set = 'Adults'
        return adults.x_train, adults.x_test, adults.y_train, adults.y_test, adults.x, adults.y
    elif args.digits:
        print('> analyzing the digits data set...')
        data_set = 'Digits'
        return digits.x_train, digits.x_test, digits.y_train, digits.y_test, digits.x, digits.y


def use_clustering_algo(k):
    # k = len(np.unique(Y))
    # print('> found {} unique labels in the data...'.format(k))
    if args.kmeans:
        print('> fitting {} clusters using k-means...'.format(k))
        # c = clustering.KMeansClustering(k)
        c = KMeans(n_clusters=k)
    if args.em:
        print('> fitting {} clusters using expectation maximization...'.format(k))
        # c = clustering.EMClustering(k)
        c = GaussianMixture(n_components=k)
    return c

def _elbow():
    # Use the elbow method to test the percent variance per number of clusters.
    pass


def cluster():
    print('Clustering the data...')
    xtrain, xtest, ytrain, ytest, X, Y = use_data_set()
    # https://pythonprogramminglanguage.com/kmeans-elbow-method/
    # https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Clustering_metrics.ipynb
    distortions = []
    sil_avgs = []
    gmm_bic = []
    num_clusters = range(2, 17)
    for k in num_clusters:
        clustering = use_clustering_algo(k)
        clustering.fit(X)
        sil_avgs.append(metrics.silhouette_score(X, clustering.predict(X)))
        if args.em: gmm_bic.append(clustering.bic(X))
        if args.kmeans: distortions.append(clustering.inertia_) # inertia is the sum of squared distances of each point to it's closest center

    # Elbow method
    if args.kmeans:
        plt.plot(num_clusters, distortions)
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Elbow method / k vs inertia: {}'.format(data_set))
        plt.tight_layout()
        plt.show()
    # Silhouette score
    plt.plot(num_clusters, sil_avgs)
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.title('Silhouette score per clustering: {}'.format(data_set))
    plt.tight_layout()
    plt.show()
    # GM BIC
    set_trace()
    if args.em:
        plt.plot(num_clusters, gmm_bic)
        plt.xlabel('k')
        plt.ylabel('Log(BIC)')
        plt.title('GMM BIC score per clustering: {}'.format(data_set))
        plt.tight_layout()
        plt.show()






























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
