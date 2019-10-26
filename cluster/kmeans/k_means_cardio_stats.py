from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from utils import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection


# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

def run_k_means_on_cardiovascular_data(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    # X, y = load_data(path + 'data/' + data_set + '/train/')

    f = open("cardiovascular_stats.txt","w+")
    bench_k_means("1", x_train, y_train, 1, f, 1)
    bench_k_means("2", x_train, y_train, 2, f, 1)
    bench_k_means("3", x_train, y_train, 3, f, 1)
    bench_k_means("4", x_train, y_train, 4, f, 1)
    bench_k_means("5", x_train, y_train, 5, f, 1)
    bench_k_means("6", x_train, y_train, 6, f, 1)
    bench_k_means("7", x_train, y_train, 7, f, 1)
    bench_k_means("8", x_train, y_train, 8, f, 1)
    bench_k_means("9", x_train, y_train, 9, f, 1)
    bench_k_means("10", x_train, y_train, 10, f, 1)
    bench_k_means("11", x_train, y_train, 11, f, 1)
    bench_k_means("12", x_train, y_train, 12, f, 1)
    bench_k_means("13", x_train, y_train, 13, f, 1)
    bench_k_means("14", x_train, y_train, 14, f, 1)
    bench_k_means("15", x_train, y_train, 15, f, 1)
    f.close()


def run_k_means_on_pca_cardiovascular_data(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    # X, y = load_data(path + 'data/' + data_set + '/train/')

    pca = PCA(n_components=5)
    pca_x_train = pca.fit_transform(x_train)

    f = open("cardiovascular_pca_stats.txt","w+")

    bench_k_means("1", pca_x_train, y_train, 1, f, 1)
    bench_k_means("2", pca_x_train, y_train, 2, f, 1)
    bench_k_means("3", pca_x_train, y_train, 3, f, 1)
    bench_k_means("4", pca_x_train, y_train, 4, f, 1)
    bench_k_means("5", pca_x_train, y_train, 5, f, 1)
    bench_k_means("6", pca_x_train, y_train, 6, f, 1)
    bench_k_means("7", pca_x_train, y_train, 7, f, 1)
    bench_k_means("8", pca_x_train, y_train, 8, f, 1)
    bench_k_means("9", pca_x_train, y_train, 9, f, 1)
    bench_k_means("10", pca_x_train, y_train, 10, f, 1)
    bench_k_means("11", pca_x_train, y_train, 11, f, 1)
    bench_k_means("12", pca_x_train, y_train, 12, f, 1)
    bench_k_means("13", pca_x_train, y_train, 13, f, 1)
    bench_k_means("14", pca_x_train, y_train, 14, f, 1)
    bench_k_means("15", pca_x_train, y_train, 15, f, 1)
    f.close()


def run_k_means_on_random_projections_cardiovascular_data(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    # X, y = load_data(path + 'data/' + data_set + '/train/')

    pca = GaussianRandomProjection(n_components=5)
    pca_x_train = pca.fit_transform(x_train)

    f = open("cardiovascular_random_projections_stats.txt","w+")

    bench_k_means("1", pca_x_train, y_train, 1, f, 1)
    bench_k_means("2", pca_x_train, y_train, 2, f, 1)
    bench_k_means("3", pca_x_train, y_train, 3, f, 1)
    bench_k_means("4", pca_x_train, y_train, 4, f, 1)
    bench_k_means("5", pca_x_train, y_train, 5, f, 1)
    bench_k_means("6", pca_x_train, y_train, 6, f, 1)
    bench_k_means("7", pca_x_train, y_train, 7, f, 1)
    bench_k_means("8", pca_x_train, y_train, 8, f, 1)
    bench_k_means("9", pca_x_train, y_train, 9, f, 1)
    bench_k_means("10", pca_x_train, y_train, 10, f, 1)
    bench_k_means("11", pca_x_train, y_train, 11, f, 1)
    bench_k_means("12", pca_x_train, y_train, 12, f, 1)
    bench_k_means("13", pca_x_train, y_train, 13, f, 1)
    bench_k_means("14", pca_x_train, y_train, 14, f, 1)
    bench_k_means("15", pca_x_train, y_train, 15, f, 1)
    f.close()


def run_k_means_on_ica_cardiovascular_data(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    # X, y = load_data(path + 'data/' + data_set + '/train/')

    pca = FastICA(n_components=5)
    pca_x_train = pca.fit_transform(x_train)

    f = open("cardiovascular_ica_stats.txt","w+")

    bench_k_means("1", pca_x_train, y_train, 1, f, 1)
    bench_k_means("2", pca_x_train, y_train, 2, f, 1)
    bench_k_means("3", pca_x_train, y_train, 3, f, 1)
    bench_k_means("4", pca_x_train, y_train, 4, f, 1)
    bench_k_means("5", pca_x_train, y_train, 5, f, 1)
    bench_k_means("6", pca_x_train, y_train, 6, f, 1)
    bench_k_means("7", pca_x_train, y_train, 7, f, 1)
    bench_k_means("8", pca_x_train, y_train, 8, f, 1)
    bench_k_means("9", pca_x_train, y_train, 9, f, 1)
    bench_k_means("10", pca_x_train, y_train, 10, f, 1)
    bench_k_means("11", pca_x_train, y_train, 11, f, 1)
    bench_k_means("12", pca_x_train, y_train, 12, f, 1)
    bench_k_means("13", pca_x_train, y_train, 13, f, 1)
    bench_k_means("14", pca_x_train, y_train, 14, f, 1)
    bench_k_means("15", pca_x_train, y_train, 15, f, 1)
    f.close()


def bench_k_means(name, data, labels, k, f, iterations):
    time_list = []
    inertia_list = []
    homogeneity_list = []

    for i in range(iterations):
        t0 = time()
        estimator = KMeans(n_clusters=k, random_state=0)
        estimator.fit(data)

        inertia_list.append(estimator.inertia_)
        homogeneity_list.append(metrics.homogeneity_score(labels, estimator.labels_))
        time_list.append(time() - t0)

    f.write('%-9s\t%.3f\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n'
            % (name,
               np.sum(time_list) / iterations,
               np.sum(inertia_list) / iterations,
               np.sum(homogeneity_list) / iterations,
               metrics.completeness_score(labels, estimator.labels_),
               metrics.v_measure_score(labels, estimator.labels_),
               metrics.adjusted_rand_score(labels, estimator.labels_),
               metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
               0.0))
    print('%-9s\t%.3f\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             0.0))


if __name__ == "__main__":
    # train_neural_net('../', False)
    # run_k_means_on_pca_cardiovascular_data('../../')
    run_k_means_on_random_projections_cardiovascular_data('../../')
    # run_k_means_on_ica_cardiovascular_data('../../')
    # run_k_means_on_cardiovascular_data('../../')
