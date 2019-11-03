import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import mean_squared_error
from time import time
from _heapq import heappush, heappop
from decomposition.decomp_util import reconstruct, time_estimator


def fa_cardio_scatter_plot(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = TruncatedSVD(n_components=5)
    pca_x_train = pca.fit_transform(x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(pca_x_train)
    print(kmeans.labels_)
    p = kmeans.predict(pca_x_train)

    for i in range(15):
        print(str(i) + " " + str(len(p[p==i])))

    h = []
    for i in range(15):
        heappush(h, (-1 * len(p[p==i]), i))
        print(str(i) + " " + str(len(p[p==i])))

    index = heappop(h)[1]
    plt.scatter(pca_x_train[:, 0][p==index].ravel(), pca_x_train[:,1][p==index].ravel(), alpha=.1, color='red')
    index = heappop(h)[1]
    plt.scatter(pca_x_train[:, 0][p==index].ravel(), pca_x_train[:,1][p==index].ravel(), alpha=.1, color='orange')
    index = heappop(h)[1]
    plt.scatter(pca_x_train[:, 0][p==index].ravel(), pca_x_train[:,1][p==index].ravel(), alpha=.1, color='blue')
    index = heappop(h)[1]
    plt.scatter(pca_x_train[:, 0][p==index].ravel(), pca_x_train[:,1][p==index].ravel(), alpha=.1, color='red')

    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('Cardiovascular Data Representation after SVD')
    # plt.show()
    plt.savefig("plots/svd_cardio.png")


def fa_loan_scatter_plot(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = TruncatedSVD(n_components=2)
    pca_x_train = pca.fit_transform(x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(pca_x_train)
    print(kmeans.labels_)
    p = kmeans.predict(pca_x_train)

    for i in range(15):
        print(str(i) + " " + str(len(p[p==i])))

    h = []
    for i in range(15):
        heappush(h, (-1 * len(p[p==i]), i))
        print(str(i) + " " + str(len(p[p==i])))

    index = heappop(h)[1]
    plt.scatter(pca_x_train[:, 0][p==index].ravel(), pca_x_train[:,1][p==index].ravel(), alpha=.1, color='red')
    index = heappop(h)[1]
    plt.scatter(pca_x_train[:, 0][p==index].ravel(), pca_x_train[:,1][p==index].ravel(), alpha=.1, color='orange')
    index = heappop(h)[1]
    plt.scatter(pca_x_train[:, 0][p==index].ravel(), pca_x_train[:,1][p==index].ravel(), alpha=.1, color='blue')

    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('Financial Loan Data Representation after SVD')
    # plt.show()
    plt.savefig("plots/svd_loan.png")


def generate_loan_svd_reconstruction_stats(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    f = open("svd_loan_reconstruction_stats.txt","w+")

    reconstruct(TruncatedSVD(n_components=1), f, x_train)
    reconstruct(TruncatedSVD(n_components=2), f, x_train)
    reconstruct(TruncatedSVD(n_components=3), f, x_train)
    reconstruct(TruncatedSVD(n_components=4), f, x_train)
    reconstruct(TruncatedSVD(n_components=5), f, x_train)
    reconstruct(TruncatedSVD(n_components=6), f, x_train)
    reconstruct(TruncatedSVD(n_components=7), f, x_train)
    reconstruct(TruncatedSVD(n_components=8), f, x_train)
    reconstruct(TruncatedSVD(n_components=9), f, x_train)
    reconstruct(TruncatedSVD(n_components=10), f, x_train)

    f.close()


def generate_cardio_svd_reconstruction_stats(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    f = open("svd_cardio_reconstruction_stats.txt","w+")

    reconstruct(TruncatedSVD(n_components=1), f, x_train)
    reconstruct(TruncatedSVD(n_components=2), f, x_train)
    reconstruct(TruncatedSVD(n_components=3), f, x_train)
    reconstruct(TruncatedSVD(n_components=4), f, x_train)
    reconstruct(TruncatedSVD(n_components=5), f, x_train)
    reconstruct(TruncatedSVD(n_components=6), f, x_train)
    reconstruct(TruncatedSVD(n_components=7), f, x_train)
    reconstruct(TruncatedSVD(n_components=8), f, x_train)
    reconstruct(TruncatedSVD(n_components=9), f, x_train)
    reconstruct(TruncatedSVD(n_components=10), f, x_train)

    f.close()


def svd_runtime_stats(path, data_set):
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    f = open("stats/svd_" + data_set + "_runtime_stats.txt","w+")

    time_estimator(TruncatedSVD(n_components=2), f, x_train)
    time_estimator(TruncatedSVD(n_components=10), f, x_train)
    time_estimator(TruncatedSVD(n_components=20), f, x_train)
    time_estimator(TruncatedSVD(n_components=30), f, x_train)
    time_estimator(TruncatedSVD(n_components=40), f, x_train)
    time_estimator(TruncatedSVD(n_components=50), f, x_train)
    time_estimator(TruncatedSVD(n_components=60), f, x_train)
    time_estimator(TruncatedSVD(n_components=70), f, x_train)
    time_estimator(TruncatedSVD(n_components=80), f, x_train)
    time_estimator(TruncatedSVD(n_components=90), f, x_train)
    time_estimator(TruncatedSVD(n_components=100), f, x_train)

    f.close()


if __name__ == "__main__":
    svd_runtime_stats('../', 'loan')
    # generate_loan_svd_reconstruction_stats('../')
    # generate_cardio_svd_reconstruction_stats('../')
    # fa_cardio_scatter_plot('../')
    # fa_loan_scatter_plot('../')
