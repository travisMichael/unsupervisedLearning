from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from utils import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score


# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
def run_k_means_on_loan_data(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    # X, y = load_data(path + 'data/' + data_set + '/train/')

    estimator = KMeans(n_clusters=15, random_state=0)
    estimator.fit(x_train)

    p = estimator.predict(x_train)

    for i in range(15):
        print(str(i) + " " + str(len(p[p==i])))

    print("%.6f" % homogeneity_score(y_train, p))

    centers = estimator.cluster_centers_

    plt.scatter(x_train[:,2][p==0].ravel(), x_train[:,3][p==0].ravel(), alpha=.1, color='red')
    plt.scatter(x_train[:,2][p==13].ravel(), x_train[:,3][p==13].ravel(), alpha=.1, color='blue')
    # plt.scatter(x_train[:,2][p==2].ravel(), x_train[:,3][p==2].ravel(), alpha=.1, color='green')
    plt.scatter(x_train[:,2][p==14].ravel(), x_train[:,3][p==14].ravel(), alpha=.1, color='yellow')
    plt.scatter(x_train[:,2][p==2].ravel(), x_train[:,3][p==2].ravel(), alpha=.1, color='green')
    plt.xlabel('Patient Height')
    plt.ylabel('Patient Weight')
    # plt.ylim((0.0, 1.0))
    plt.show()

    plt.xlabel('Patient Height')
    plt.ylabel('Patient Weight')

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
