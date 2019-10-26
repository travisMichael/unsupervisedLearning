import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import mean_squared_error
from time import time
from _heapq import heappush, heappop


def fa_cardio_scatter_plot(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = FactorAnalysis(n_components=5)
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

    plt.xlabel('FA Component 1')
    plt.ylabel('FA Component 2')
    plt.title('Cardiovascular Data Representation after FA')
    # plt.show()
    plt.savefig("plots/fa_cardio.png")


def fa_loan_scatter_plot(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = FactorAnalysis(n_components=2)
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

    plt.xlabel('FA Component 1')
    plt.ylabel('FA Component 2')
    plt.title('Financial Loan Data Representation after FA')
    # plt.show()
    plt.savefig("plots/fa_loan.png")


if __name__ == "__main__":
    # fa_cardio_scatter_plot('../')
    fa_loan_scatter_plot('../')
