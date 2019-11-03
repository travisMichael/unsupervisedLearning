import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import mean_squared_error
from time import time
from _heapq import heappush, heappop
from decomposition.decomp_util import reconstruct, time_estimator


def pca_cardio_scatter_plot(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = PCA(n_components=5)
    pca_x_train = pca.fit_transform(x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(pca_x_train)
    print(kmeans.labels_)
    p = kmeans.predict(pca_x_train)

    for i in range(15):
        print(str(i) + " " + str(len(p[p==i])))

    plt.scatter(pca_x_train[:, 0][p==0].ravel(), pca_x_train[:,1][p==0].ravel(), alpha=.1, color='yellow')
    plt.scatter(pca_x_train[:, 0][p==6].ravel(), pca_x_train[:,1][p==6].ravel(), alpha=.1, color='orange')
    plt.scatter(pca_x_train[:, 0][p==2].ravel(), pca_x_train[:,1][p==2].ravel(), alpha=.1, color='blue')
    plt.scatter(pca_x_train[:, 0][p==7].ravel(), pca_x_train[:,1][p==7].ravel(), alpha=.1, color='red')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Cardiovascular Data Representation after PCA')
    # plt.show()
    plt.savefig("plots/pca_cardio.png")


def pca_loan_scatter_plot(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = PCA(n_components=2)
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


    # plt.scatter(pca_x_train[:, 0][p==0].ravel(), pca_x_train[:,1][p==0].ravel(), alpha=.1, color='yellow')
    # plt.scatter(pca_x_train[:, 0][p==6].ravel(), pca_x_train[:,1][p==6].ravel(), alpha=.1, color='orange')
    # plt.scatter(pca_x_train[:, 0][p==2].ravel(), pca_x_train[:,1][p==2].ravel(), alpha=.1, color='blue')
    # plt.scatter(pca_x_train[:, 0][p==7].ravel(), pca_x_train[:,1][p==7].ravel(), alpha=.1, color='red')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Financial Loan Data Representation after PCA')
    # plt.show()
    plt.savefig("plots/pca_loan.png")


def pca_runtime_stats(path, data_set):
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    f = open("stats/pca_" + data_set + "_runtime_stats.txt","w+")

    time_estimator(PCA(n_components=2), f, x_train)
    time_estimator(PCA(n_components=10), f, x_train)
    time_estimator(PCA(n_components=20), f, x_train)
    time_estimator(PCA(n_components=30), f, x_train)
    time_estimator(PCA(n_components=40), f, x_train)
    time_estimator(PCA(n_components=50), f, x_train)
    time_estimator(PCA(n_components=60), f, x_train)
    time_estimator(PCA(n_components=70), f, x_train)
    time_estimator(PCA(n_components=80), f, x_train)
    time_estimator(PCA(n_components=90), f, x_train)
    time_estimator(PCA(n_components=100), f, x_train)

    f.close()


def generate_loan_pca_reconstruction_stats(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    f = open("pca_loan_reconstruction_stats.txt","w+")

    reconstruct(PCA(n_components=1), f, x_train)
    reconstruct(PCA(n_components=2), f, x_train)
    reconstruct(PCA(n_components=3), f, x_train)
    reconstruct(PCA(n_components=4), f, x_train)
    reconstruct(PCA(n_components=5), f, x_train)
    reconstruct(PCA(n_components=6), f, x_train)
    reconstruct(PCA(n_components=7), f, x_train)
    reconstruct(PCA(n_components=8), f, x_train)
    reconstruct(PCA(n_components=9), f, x_train)
    reconstruct(PCA(n_components=10), f, x_train)

    f.close()


def generate_cardio_pca_reconstruction_stats(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    f = open("pca_cardio_reconstruction_stats.txt","w+")

    reconstruct(PCA(n_components=1), f, x_train)
    reconstruct(PCA(n_components=2), f, x_train)
    reconstruct(PCA(n_components=3), f, x_train)
    reconstruct(PCA(n_components=4), f, x_train)
    reconstruct(PCA(n_components=5), f, x_train)
    reconstruct(PCA(n_components=6), f, x_train)
    reconstruct(PCA(n_components=7), f, x_train)
    reconstruct(PCA(n_components=8), f, x_train)
    reconstruct(PCA(n_components=9), f, x_train)
    reconstruct(PCA(n_components=10), f, x_train)

    f.close()


def generate_pca_variance(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = PCA(n_components=10)
    pca_x_train = pca.fit_transform(x_train)

    # plt.hist(pca.explained_variance_, 5, facecolor='black', alpha=0.5)

    features = range(pca.n_components_)
    plt.show()
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.title('PCA - Explained Variance Ratio for \nFinancial Loan Dataset (Standard)')
    # plt.show()
    plt.savefig("plots/pca_variance_loan_standard.png")
    print()


if __name__ == "__main__":
    pca_runtime_stats('../', 'loan')
    # generate_pca_variance('../')
    # generate_cardio_pca_reconstruction_stats('../')
    # generate_loan_pca_reconstruction_stats('../')
    # pca_cardio_scatter_plot('../')
    # pca_loan_scatter_plot('../')
