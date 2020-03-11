
# sklearn calls expectation maximization "gaussian mixture models"
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
        k = len(np.unique(y))
        print(np.unique(y))
        print('> discovered {} unique labels in the data'.format(k))
        self.init(len(np.unique(y)))
        start = time.time()
        result = self.clustering().fit(x, y)
        end = time.time()
        print('\tfitting data took: {:.5f} secs'.format(end - start))
        return result

    def test(self, x):
        return self.clustering().predict(x)


class KMeansClustering(AbstractClustering):
    def init(self, k):
        self.k = k
        self.kmeans = KMeans(n_clusters=k)

    def clustering(self):
        return self.kmeans

    def experiment(self, xtrain, xtest, ytrain, ytest, X, Y):
        self.train(X, Y)
        unique = np.unique(Y)
        labels = self.clustering().predict(X)
        # plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis')
        # plt.show()
        pca = PCA(n_components=self.k).fit_transform(X)
        serieses = []
        for i in range(0, pca.shape[0]):
            print(i)
            series = plt.text(pca[i,0], pca[i,1], str(unique[i]))
            serieses.append(series)
        plt.legend(serieses, unique)
        # plt.scatter(pca[:,0], pca[:,1], c=labels)
        plt.show()


class EMClustering(AbstractClustering):

    def init(self, k):
        self.k = k
        self.em = GaussianMixture(n_components=k)

    def clustering(self):
        return self.em

    def experiment(self, xtrain, xtest, ytrain, ytest):
        pass

