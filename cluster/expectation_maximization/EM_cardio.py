from time import time
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from utils import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score


# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
def run_k_means_on_loan_data(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    estimator = GaussianMixture(n_components=3, random_state=0)
    estimator.fit(x_train)

    predictions = estimator.predict(x_train)
    ss = metrics.silhouette_score(x_train, predictions, metric='euclidean', sample_size=300)
    print("%.6f" % homogeneity_score(y_train, predictions))
    print("%.6f" % ss)

    estimator = KMeans(n_clusters=3, random_state=0)
    estimator.fit(x_train)

    predictions = estimator.predict(x_train)
    ss = metrics.silhouette_score(x_train, estimator.labels_, metric='euclidean', sample_size=300)

    print("%.6f" % homogeneity_score(y_train, predictions))
    print("%.6f" % ss)
    ss = metrics.silhouette_score(x_train, predictions, metric='euclidean', sample_size=300)
    print("%.6f" % ss)

    # plt.scatter(x_train[:,92], x_train[:,93], alpha=.1, color='black')
    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')
    # plt.ylim((-1.0, 1.0))
    # plt.show()

    # f.close()


if __name__ == "__main__":
    # train_neural_net('../', False)
    run_k_means_on_loan_data('../../')
