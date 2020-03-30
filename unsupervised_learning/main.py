
import sys
from data import adults, digits, split
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
import time
#
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
#
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
from scipy.stats import kurtosis
from scipy.linalg import pinv # pseudo inverse function of matrix
from scipy.sparse import issparse
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py

data_set = None

def use_data_set():
    global data_set
    if args.adults:
        print('> analyzing the adults data set...')
        data_set = 'Adults'
        return adults.x, adults.y
    elif args.digits:
        print('> analyzing the digits data set...')
        data_set = 'Digits'
        return digits.x, digits.y


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


def use_reduction_algo(m):
    if args.pca:
        print('> reducing to {} features using PCA...'.format(m))
        reducer = PCA(n_components=m)
    if args.ica:
        print('> reducing to {} features using ICA...'.format(m))
        reducer = FastICA(n_components=m)
    if args.rca:
        print('> reducing to {} features using Randomized Projections...'.format(m))
        # reducer = SparseRandomProjection(n_components=m)
        reducer = GaussianRandomProjection(n_components=m)
    if args.svd:
        print('> reducing to {} features using Singular Value Decomp...'.format(m))
        reducer = TruncatedSVD(n_components=m)
    return reducer


def cluster(X, Y=None):
    print('Clustering the data...')
    # https://pythonprogramminglanguage.com/kmeans-elbow-method/
    # https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Clustering_metrics.ipynb
    distortions = []
    sil_avgs = []
    gmm_bic = []
    scores = []
    num_clusters = range(2, 17)
    for k in num_clusters:
        clustering = use_clustering_algo(k)
        clustering.fit(X)
        sil_avgs.append(silhouette_score(X, clustering.predict(X)))
        scores.append(clustering.score(X))
        if args.em: gmm_bic.append(-1 * clustering.bic(X))
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
    # set_trace()
    if args.em:
        plt.plot(num_clusters, np.gradient(gmm_bic))
        plt.xlabel('k')
        plt.ylabel('gradient(BIC)')
        plt.title('GMM BIC score per clustering: {}'.format(data_set))
        plt.tight_layout()
        plt.show()
    # Scores
    plt.plot(num_clusters, scores)
    plt.xlabel('k')
    plt.ylabel('Clustering score')
    plt.title('Score per clustering: {}'.format(data_set))
    plt.tight_layout()
    plt.show()


def dim_reduce(X, Y):
    # https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
    print('Reducing the dimensions of the data...')
    # xtrain, xtest, ytrain, ytest, X, Y = use_data_set()
    dims = range(2, 63)
    pca_total_var = []
    lda_total_var = []
    ica_kurtosis_means = []
    rca_recon_errors = []
    pca_recon_errors = []
    ica_recon_errors = []
    kpca_recon_errors = {}
    for m in dims:
        print('Using {} components...'.format(m))
        if args.pca:
            pca = PCA(n_components=m)
            X_pca = pca.fit_transform(X)
            total_variance = np.sum(pca.explained_variance_ratio_)
            print(' > PCA total variance: {}'.format(total_variance))
            pca_total_var.append(total_variance)
            X_recon = pca.inverse_transform(X_pca)
            pca_recon_errors.append(np.square(np.subtract(X, X_recon)).mean())
        if args.ica:
            ica = FastICA(n_components=m)
            X_ica = ica.fit_transform(X)
            kurtosis_mean = np.mean(np.abs(kurtosis(X_ica)))
            print(' > ICA kurtosis mean: {}'.format(kurtosis_mean))
            ica_kurtosis_means.append(kurtosis_mean)
            X_recon = ica.inverse_transform(X_ica)
            ica_recon_errors.append(np.square(np.subtract(X, X_recon)).mean())
        if args.rca:
            rca = GaussianRandomProjection(n_components=m)
            X_rca = rca.fit_transform(X)
            # calculate the reconstruction error
            # https://omscs-study.slack.com/archives/C08LK14DV/p1584134696113900
            w = rca.components_ # random matrix used in the projection
            w_inv = pinv(w.T) # find the pseudo inverse of the projection matrix
            X_recon = X_rca @ w_inv
            rca_recon_errors.append(np.square(np.subtract(X, X_recon)).mean())
        if args.kpca:
            # for kernel in ['linear', 'rbf', 'cosine']:
            for kernel in ['linear']:
                kpca = KernelPCA(n_components=m, fit_inverse_transform=True, kernel=kernel)
                X_kpca = kpca.fit_transform(X)
                # https://stackoverflow.com/questions/29611842/scikit-learn-kernel-pca-explained-variance
                # explained_variance = np.var(X_kpca, axis=0)
                # explained_variance_ratio = explained_variance / np.sum(explained_variance)
                # kpca_total_var.append(np.sum(explained_variance_ratio))
                # total_variance = np.sum(kpca.explained_variance_ratio_)
                # print(' > Kernel-PCA total variance: {}'.format(total_variance))
                # pca_total_var.append(total_variance)
                X_recon = kpca.inverse_transform(X_kpca)
                kpca_recon_error = np.square(np.subtract(X, X_recon)).mean()
                if kernel in kpca_recon_errors.keys():
                    kpca_recon_errors[kernel].append(kpca_recon_error)
                else:
                    kpca_recon_errors[kernel] = [kpca_recon_error]
        if args.lda:
            lda = LinearDiscriminantAnalysis(n_components=m)
            X_lda = lda.fit_transform(X, Y)
            lda_total_var.append(np.sum(lda.explained_variance_ratio_))

            # set_trace()
    # Test multiple methods against each other
    # Scores
    # set_trace()
    # https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe
    if args.pca:
        plt.plot(dims, pca_total_var)
        plt.xlabel('No. components'), plt.ylabel('Variance (%)')
        plt.title('Preserved variance per no. components: {}'.format(data_set))
        plt.tight_layout()
        plt.show()
    if args.ica:
        # https://piazza.com/class/k51r1vdohil5g3?cid=592_f11
        plt.plot(dims, ica_kurtosis_means)
        plt.xlabel('No. components'), plt.ylabel('Kurtosis')
        plt.title('Kurtosis per no. components: {}'.format(data_set))
        plt.tight_layout()
        plt.show()
    if args.rca:
        # You want to find the error from the original and reconstructed data
        # To do this, you need the pseudo inverse of the projection matrix
        # Use inverse_transform somwhere...
        # https://piazza.com/class/k51r1vdohil5g3?cid=592_f15
        plt.plot(dims, rca_recon_errors)
        plt.xlabel('No. components'), plt.ylabel('Reconstruction error')
        plt.title('Reconstruction error per no. components: {}'.format(data_set))
        plt.tight_layout()
        plt.show()
    if args.pca and args.lda:
        plt.plot(dims, pca_total_var, label='PCA')
        plt.plot(dims, lda_total_var, label='LDA')
        plt.xlabel('No. components'), plt.ylabel('Variance (%)')
        plt.title('Preserved variance per no. components: {}'.format(data_set))
        plt.tight_layout(),  plt.legend(loc="best")
        plt.show()
    if args.pca and args.rca and args.kpca:
        plt.plot(dims, pca_recon_errors, label='PCA')
        plt.plot(dims, rca_recon_errors, label='RCA')
        for kernel, recon_errors in kpca_recon_errors.items():
            plt.plot(dims, recon_errors, label='KPCA [{}]'.format(kernel))
        plt.xlabel('No. components'), plt.ylabel('Reconstruction error')
        plt.title('Reconstruction error per no. components: {}'.format(data_set))
        plt.tight_layout(), plt.legend(loc="best")
        plt.show()
    if args.lda:
        pass



def cluster_and_reduce(X, Y):
    print('Reducing then clustering the data...')
    # xtrain, xtest, ytrain, ytest, X, Y = use_data_set()
    num_clusters = range(2, 15)
    # dims = [10, 15, 20, 25, 30, 40, 50, 55, 60]
    dims = [50]
    # visualizations of all data in two dimensions
    if args.visualize and args.kmeans:
        m = 2
        pca = PCA(n_components=m)
        ica = FastICA(n_components=m)
        rca = GaussianRandomProjection(n_components=m)
        lda = LinearDiscriminantAnalysis(n_components=m)
        #
        start = time.time()
        X_pca = pca.fit_transform(X)
        print('PCA fit took {} seconds...'.format(time.time() - start))
        start = time.time()
        X_ica = ica.fit_transform(X)
        print('ICA fit took {} seconds...'.format(time.time() - start))
        start = time.time()
        X_rca = rca.fit_transform(X)
        print('RCA fit took {} seconds...'.format(time.time() - start))
        start = time.time()
        X_lda = lda.fit_transform(X, Y)
        print('LDA fit took {} seconds...'.format(time.time() - start))
        #
        k = 3
        clustering = use_clustering_algo(k)
        pca_labels = clustering.fit_predict(X_pca)
        ica_labels = clustering.fit_predict(X_ica)
        rca_labels = clustering.fit_predict(X_rca)
        lda_labels = clustering.fit_predict(X_lda)
        plt.scatter(X_pca[:,0], X_pca[:,1], c=pca_labels, cmap='viridis', label='PCA'), plt.show()
        plt.scatter(X_ica[:,0], X_ica[:,1], c=ica_labels, cmap='viridis', label='ICA'), plt.show()
        plt.scatter(X_rca[:,0], X_rca[:,1], c=rca_labels, cmap='viridis', label='RCA'), plt.show()
        # plt.scatter(X_lda[:,0], X_lda[:,1], c=lda_labels, cmap='viridis', label='LDA'), plt.show()
    if args.visualize and args.em:
        m = 2
        pca = PCA(n_components=m)
        ica = FastICA(n_components=m)
        rca = GaussianRandomProjection(n_components=m)
        lda = LinearDiscriminantAnalysis(n_components=m)
        #
        X_pca = pca.fit_transform(X)
        X_ica = ica.fit_transform(X)
        X_rca = rca.fit_transform(X)
        X_lda = lda.fit_transform(X, Y)
        #
        k = 4
        clustering = use_clustering_algo(k)
        pca_labels = clustering.fit_predict(X_pca)
        ica_labels = clustering.fit_predict(X_ica)
        rca_labels = clustering.fit_predict(X_rca)
        lda_labels = clustering.fit_predict(X_lda)
        plt.scatter(X_pca[:,0], X_pca[:,1], c=pca_labels, cmap='viridis', label='PCA'), plt.show()
        plt.scatter(X_ica[:,0], X_ica[:,1], c=ica_labels, cmap='viridis', label='ICA'), plt.show()
        plt.scatter(X_rca[:,0], X_rca[:,1], c=rca_labels, cmap='viridis', label='RCA'), plt.show()
        # plt.scatter(X_lda[:,0], X_lda[:,1], c=lda_labels, cmap='viridis', label='LDA'), plt.show()


    if not args.visualize:
        for m in dims:
            print('Using {} components...'.format(m))
            pca = PCA(n_components=m)
            ica = FastICA(n_components=m)
            rca = GaussianRandomProjection(n_components=m)
            lda = LinearDiscriminantAnalysis(n_components=m)
            #
            start = time.time()
            X_pca = pca.fit_transform(X)
            print('PCA time to fit {} features: {} seconds...'.format(m, time.time() - start))
            start = time.time()
            X_ica = ica.fit_transform(X)
            print('ICA time to fit {} features: {} seconds...'.format(m, time.time() - start))
            start = time.time()
            X_rca = rca.fit_transform(X)
            print('RCA time to fit {} features: {} seconds...'.format(m, time.time() - start))
            start = time.time()
            X_lda = lda.fit_transform(X, Y)
            print('LDA time to fit {} features: {} seconds...'.format(m, time.time() - start))
            #
            distortions = {'PCA': [], 'ICA': [], 'RCA': [], 'LDA': []}
            silhouettes = {'PCA': [], 'ICA': [], 'RCA': [], 'LDA': []}
            gmm_bics = {'PCA': [], 'ICA': [], 'RCA': [], 'LDA': []}

            for k in num_clusters:
                clustering = use_clustering_algo(k)
                clustering.fit(X_pca)
                if args.kmeans: 
                    silhouettes['PCA'].append(silhouette_score(X_pca, clustering.predict(X_pca)))
                    distortions['PCA'].append(clustering.inertia_)
                if args.em: gmm_bics['PCA'].append(-1 * clustering.bic(X_pca))

                clustering.fit(X_ica)
                if args.kmeans: 
                    distortions['ICA'].append(clustering.inertia_)
                    silhouettes['ICA'].append(silhouette_score(X_ica, clustering.predict(X_ica)))
                if args.em: gmm_bics['ICA'].append(-1 * clustering.bic(X_ica))


                clustering.fit(X_rca)
                if args.kmeans: 
                    distortions['RCA'].append(clustering.inertia_)
                    silhouettes['RCA'].append(silhouette_score(X_rca, clustering.predict(X_rca)))
                if args.em: gmm_bics['RCA'].append(-1 * clustering.bic(X_rca))


                clustering.fit(X_lda)
                if args.kmeans: 
                    distortions['LDA'].append(clustering.inertia_)
                    silhouettes['LDA'].append(silhouette_score(X_lda, clustering.predict(X_lda)))
                if args.em: gmm_bics['LDA'].append(-1 * clustering.bic(X_lda))


            if args.kmeans:
                # Elbow inertia
                # for dimred, distortions in distortions.items():
                #     plt.plot(num_clusters, distortions, label=dimred)
                # plt.xlabel('k'), plt.ylabel('inertia'), plt.legend(loc='best')
                # plt.title('Elbow method / k vs inertia: {}'.format(data_set))
                # plt.tight_layout(), plt.show()
                # Silhouette scores
                for dimred, sils in silhouettes.items():
                    plt.plot(num_clusters, sils, label=dimred)
                plt.xlabel('k'), plt.ylabel('Silhouette score'), plt.legend(loc='best')
                plt.title('Silhouette score per clustering: {} [m={}]'.format(data_set, m))
                plt.tight_layout(), plt.show()
            if args.em:
                for dimred, bics in gmm_bics.items():
                    plt.plot(num_clusters, np.gradient(bics), label=dimred)
                plt.xlabel('k'), plt.ylabel('gradient(BIC)'), plt.legend(loc='best')
                plt.title('GMM BIC score per clustering: {} [m={}]'.format(data_set, m))
                plt.tight_layout(), plt.show()


def reduce_and_nn(X, Y):
    print('Applying dimension reduction and running the neural network...')
    dims = [10, 30, 50]
    for m in dims:
        print('Using {} components...'.format(m))
        nn = MLPClassifier(hidden_layer_sizes=(16, 16, 16))
        nn_pca = MLPClassifier(hidden_layer_sizes=(16, 16, 16))
        nn_ica = MLPClassifier(hidden_layer_sizes=(16, 16, 16))
        nn_rca = MLPClassifier(hidden_layer_sizes=(16, 16, 16))
        nn_lda = MLPClassifier(hidden_layer_sizes=(16, 16, 16))
        #
        # nn_pca.fit(PCA(n_components=m).fit_transform(X), Y)
        # nn_ica.fit(FastICA(n_components=m).fit_transform(X), Y)
        # nn_rca.fit(GaussianRandomProjection(n_components=m).fit_transform(X), Y)
        # nn_lda.fit(LinearDiscriminantAnalysis(n_components=m).fit_transform(X, Y), Y)
        # pca_xtrain, pca_xtest, pca_ytrain, pca_ytest = split(PCA(n_components=m).fit_transform(X), Y)
        # ica_xtrain, ica_xtest, ica_ytrain, ica_ytest = split(FastICA(n_components=m).fit_transform(X), Y)
        # rca_xtrain, rca_xtest, rca_ytrain, rca_ytest = split(GaussianRandomProjection(n_components=m).fit_transform(X), Y)
        # lda_xtrain, lda_xtest, lda_ytrain, lda_ytest = split(LinearDiscriminantAnalysis(n_components=m).fit_transform(X, Y), Y)
        # 
        # nn_pca.fit(pca_xtrain, pca_ytrain)
        # nn_ica.fit(ica_xtrain, ica_ytrain)
        # nn_rca.fit(rca_xtrain, rca_ytrain)
        # nn_lda.fit(lda_xtrain, lda_ytrain)
        #
        plot_learning_curve(nn, X, Y, 'default', 'blue', 'navy')
        if args.pca: plot_learning_curve(nn_pca, PCA(n_components=m).fit_transform(X), Y, 'PCA', 'green', 'darkgreen')
        if args.ica: plot_learning_curve(nn_ica, FastICA(n_components=m).fit_transform(X), Y, 'ICA', 'violet', 'darkviolet')
        if args.rca: plot_learning_curve(nn_rca, GaussianRandomProjection(n_components=m).fit_transform(X), Y, 'RCA', 'cyan', 'darkcyan')
        if args.lda: plot_learning_curve(nn_lda, LinearDiscriminantAnalysis(n_components=m).fit_transform(X, Y), Y, 'LCA', 'sandybrown', 'saddlebrown')
        plt.title('Learning Curve: {} [m={}]'.format(data_set, m))
        plt.show()


def plot_learning_curve(nn, X, Y, lbl, trcolor='darkorange', tecolor='navy', plot_cv=True):
    scoring = 'accuracy'
    train_sizes, train_scores, test_scores, = learning_curve(
        nn, 
        X, 
        Y,
        scoring=scoring,
        cv=5)
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, color=trcolor,  label="[{}] training score".format(lbl))
    if plot_cv: plt.plot(train_sizes, test_mean, color=tecolor, label="[{}] validation score".format(lbl))
    # plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color=trcolor)
    if plot_cv: plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color=tecolor)
    plt.xlabel("Training Set Size"), plt.ylabel(scoring), plt.legend(loc="best")


def cluster_and_nn(X, Y):
    print('Clustering data and running the neural network...')
    if args.adults:
        k = 2
    if args.digits:
        k = 10
    print('Using {} clusters...'.format(k))
    # X = xtest
    # Y = ytest
    km = KMeans(n_clusters=k)
    em = GaussianMixture(n_components=k)
    #
    km_clusters = km.fit_predict(X)
    # km.fit(xtrain)
    em_clusters = em.fit_predict(X)
    # em.fit(xtrain)
    # set_trace()
    #
    nn = MLPClassifier(hidden_layer_sizes=(50, 50, 50))
    nn_kmeans = MLPClassifier(hidden_layer_sizes=(50, 50, 50))
    nn_em = MLPClassifier(hidden_layer_sizes=(50, 50, 50))
    #
    plot_learning_curve(nn, X, Y, 'control', 'blue', 'navy')
    #nn_kmeans.fit(KMeans(n_clusters=k).)
    if args.kmeans: plot_learning_curve(nn_kmeans, km_clusters.reshape(-1, 1), Y, 'k-means', 'green', 'darkgreen')
    # em doesn't include a transform method so we use the predict_proba instead
    # https://github.com/scikit-learn/scikit-learn/issues/7743
    if args.em: plot_learning_curve(nn_em, em_clusters.reshape(-1, 1), Y, 'em', 'violet', 'darkviolet')
    plt.title('Learning Curve: {} [k={}]'.format(data_set, k))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select an experiment to run')
    parser.add_argument('--kmeans', action='store_true', help='Use the k-means clustering algorithm')
    parser.add_argument('--em', action='store_true', help='Use the expectation maximization algorithm')

    parser.add_argument('--pca', action='store_true', help='Reduce dimensions using PCA')
    parser.add_argument('--ica', action='store_true', help='Reduce dimensions using ICA')
    parser.add_argument('--rca', action='store_true', help='Reduce dimensions using randomized projections')
    parser.add_argument('--kpca', action='store_true', help='Reduce dimensions using Kernel-PCA')
    parser.add_argument('--lda', action='store_true', help='Reduce dimensions using LDA')

    parser.add_argument('--nn', action='store_true', help='Run the neural network on the resultant data')
    
    parser.add_argument('--visualize', action='store_true', help='Visualize the clusters')

    parser.add_argument('--adults', action='store_true', help='Experiment with the U.S. adults data set')
    parser.add_argument('--digits', action='store_true', help='Experiment with the handwritten digits data set')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
    do_clustering = args.kmeans or args.em
    do_reduction = args.pca or args.ica or args.rca or args.kpca
    do_nn = args.nn

    X, Y = use_data_set()

    if do_clustering and not do_reduction and not do_nn:
        # Only do clustering
        cluster(X, Y)
    if do_reduction and not do_clustering and not do_nn:
        # Only do reduction
        dim_reduce(X, Y)
    if do_clustering and do_reduction and not do_nn:
        # Do both clustering and reduction
        cluster_and_reduce(X, Y)
    if do_reduction and do_nn and not do_clustering:
        # Run a nn on the reduced dimensions
        reduce_and_nn(X, Y)
    if do_clustering and do_nn and not do_reduction:
        # Run a nn on the clusted data as if it were dim reduction
        cluster_and_nn(X, Y)
