from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from utils import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score


# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
def run_k_means_on_loan_data(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    # X, y = load_data(path + 'data/' + data_set + '/train/')

    estimator = KMeans(n_clusters=3, random_state=0)
    estimator.fit(x_train)

    predictions = estimator.predict(x_train)

    print("%.6f" % homogeneity_score(y_train, predictions))

    centers = estimator.cluster_centers_

    zeros = np.zeros(x_train.shape)
    for i in range(len(x_train)):
        zeros[i] = (centers[0] - x_train[i]) ** 2

    zeros_1 = np.zeros(x_train.shape)
    for i in range(len(x_train)):
        zeros_1[i] = (centers[1] - x_train[i]) ** 2

    # d_1 = (centers[0][0] - x_train[0][0]) ** 2 + (centers[0][1] - x_train[0][1]) ** 2

    d = zeros + zeros_1

    d = np.sqrt(d)

    d_0 = d[predictions[predictions==0].ravel()]
    d_1 = d[predictions[predictions==1].ravel()]
    d_2 = d[predictions[predictions==2].ravel()]

    d = d.T

    d = np.sum(d , axis=1)

    sort_args = np.argsort(d)

    plt.scatter(x_train[:,92], x_train[:,93], alpha=.1, color='black')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.ylim((-1.0, 1.0))
    plt.show()
    #
    # plt.scatter(x_train[:,10], x_train[:,11], alpha=.1, color='black')
    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')
    # plt.ylim((-1.0, 1.0))
    # plt.show()

    plt.scatter(x_train[:,3], x_train[:,4], alpha=.1, color='black')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.ylim((-1.0, 3.0))
    plt.show()

    # plt.scatter(x_train[:,2], x_train[:,3], alpha=.1, color='black')
    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')
    # plt.ylim((-1.0, 1.0))
    # plt.show()

    plt.scatter(x_train[:,1], x_train[:,2], alpha=.1, color='black')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.ylim((-1.0, 1.0))
    plt.show()

    print(centers)

    # f = open("loan_stats.txt","w+")
    # bench_k_means("1", x_train, y_train, 1, f)
    # bench_k_means("2", x_train, y_train, 2, f)
    # bench_k_means("3", x_train, y_train, 3, f)
    # bench_k_means("4", x_train, y_train, 4, f)
    # bench_k_means("5", x_train, y_train, 5, f)
    # bench_k_means("6", x_train, y_train, 6, f)
    # bench_k_means("7", x_train, y_train, 7, f)
    # bench_k_means("8", x_train, y_train, 8, f)
    # bench_k_means("9", x_train, y_train, 9, f)
    # bench_k_means("10", x_train, y_train, 10, f)
    # bench_k_means("11", x_train, y_train, 11, f)
    # bench_k_means("12", x_train, y_train, 12, f)
    # bench_k_means("13", x_train, y_train, 13, f)
    # bench_k_means("14", x_train, y_train, 14, f)
    # bench_k_means("15", x_train, y_train, 15, f)
    # f.close()


def run_k_means_on_cardiovascular_data(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    # X, y = load_data(path + 'data/' + data_set + '/train/')

    f = open("cardiovascular_stats.txt","w+")
    bench_k_means("1", x_train, y_train, 1, f)
    bench_k_means("2", x_train, y_train, 2, f)
    bench_k_means("3", x_train, y_train, 3, f)
    bench_k_means("4", x_train, y_train, 4, f)
    bench_k_means("5", x_train, y_train, 5, f)
    bench_k_means("6", x_train, y_train, 6, f)
    bench_k_means("7", x_train, y_train, 7, f)
    bench_k_means("8", x_train, y_train, 8, f)
    bench_k_means("9", x_train, y_train, 9, f)
    bench_k_means("10", x_train, y_train, 10, f)
    bench_k_means("11", x_train, y_train, 11, f)
    bench_k_means("12", x_train, y_train, 12, f)
    bench_k_means("13", x_train, y_train, 13, f)
    bench_k_means("14", x_train, y_train, 14, f)
    bench_k_means("15", x_train, y_train, 15, f)
    f.close()


def bench_k_means(name, data, labels, k, f):
    t0 = time()
    estimator = KMeans(n_clusters=k, random_state=0)
    estimator.fit(data)
    s = 0.0
    # if k > 1:
    #     s = metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=300)

    f.write('%-9s\t%.3f\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n'
            % (name, (time() - t0), estimator.inertia_,
               metrics.homogeneity_score(labels, estimator.labels_),
               metrics.completeness_score(labels, estimator.labels_),
               metrics.v_measure_score(labels, estimator.labels_),
               metrics.adjusted_rand_score(labels, estimator.labels_),
               metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
               s))
    print('%-9s\t%.3f\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             s))


if __name__ == "__main__":
    # train_neural_net('../', False)
    run_k_means_on_loan_data('../../')
