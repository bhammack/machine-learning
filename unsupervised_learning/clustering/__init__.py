
# sklearn calls expectation maximization "gaussian mixture models"
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pdb import set_trace


from abc import ABC, abstractmethod

# Wow that's nice...
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html?highlight=gaussianmixture#sklearn.mixture.GaussianMixture

class AbstractClustering():

    def __init__(self):
        self.k = None

    @abstractmethod
    def init(self, k):
        """Inits the instance of the clustering algorithm."""
        pass

    @abstractmethod
    def clustering(self):
        """Returns the instance of the clustering algorithm."""
        pass

    def get_params(self):
        return self.clustering().get_params()

    def set_params(self, **params):
        return self.clustering().set_params(**params)

    def probability(self, x):
        return self.clustering().predict_proba(x)

    def experiment(self, xtrain, xtest, ytrain, ytest, X, Y):
        """Use a custom experiment defined by the learner."""
        print('No experiment defined!')

    def train(self, x, y):
        # k = len(np.unique(y))
        # print(np.unique(y))
        # print('> discovered {} unique labels in the data'.format(k))
        # self.init(len(np.unique(y)))
        start = time.time()
        result = self.clustering().fit(x, y)
        end = time.time()
        print('\tfitting data took: {:.5f} secs'.format(end - start))
        return result

    def test(self, x):
        return self.clustering().predict(x)


class KMeansClustering(AbstractClustering):
    def __init__(self, k):
        self.k = k
        self.kmeans = KMeans(n_clusters=k)

    def clustering(self):
        return self.kmeans

    def experiment(self, xtrain, xtest, ytrain, ytest, X, Y):
        # self.train(X, Y)
        # self.train(X, Y)
        # k = len(np.unique(Y))
        # self.clustering().set_params(n_clusters=k)
        # 1. Reduce the data
        # 2. Fit to the clustering algorithm
        # 3. Graph



        # Test fitting on the raw data, no PCA
        self.clustering().fit(X)
        labels = self.clustering().predict(X)
        # Since we're not using PCA, we have a lot of dimensions to consider when plotting a clustering...
        set_trace()
        # age vs hours: bad
        # age vs gains: bad
        # age vs education: pretty uniform distribution
        # gains vs loss: oddly L shaped....
        # age vs fnlwgt: interesting decision boundary!
        # education vs gains: eh...
        # education vs hours: eh...
        # white vs male - interesting, but binary so I can't visualize clusters well...
        # fnlwgt vs gains: almost IDENTICAL to clustering as PCA reduced...
        # plt.scatter(X[2], X[10], c=labels, cmap='viridis')
        plt.scatter(X['8_ White'], X['9_ Male'], c=labels, cmap='viridis')
        centroids = self.clustering().cluster_centers_
        # plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, alpha=0.5)
        plt.show()




        # Test fitting on the PCA reduced data
        pca = PCA(n_components=self.k)
        X_2d = pca.fit_transform(X)
        self.clustering().fit(X_2d)
        labels = self.clustering().predict(X_2d)
        # https://stackoverflow.com/questions/31150982/pca-output-looks-weird-for-a-kmeans-scatter-plot
        # over 99.9% variance captured by 2d data
        print(pca.explained_variance_ratio_)
        plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='viridis')
        centroids = self.clustering().cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, alpha=0.5)
        plt.show()


class EMClustering(AbstractClustering):

    def __init__(self, k):
        self.k = k
        self.em = GaussianMixture(n_components=k)

    def clustering(self):
        return self.em

    def experiment(self, xtrain, xtest, ytrain, ytest):
        pass

