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


def run_stats(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    f = open("pca_runtime_stats.txt","w+")

    run(1, f, x_train)
    run(2, f, x_train)
    run(3, f, x_train)
    run(4, f, x_train)
    run(5, f, x_train)
    run(6, f, x_train)
    run(8, f, x_train)
    run(9, f, x_train)
    run(10, f, x_train)

    f.close()


def run(n, f, train):
    t = 0.0
    a = 20
    for i in range(a):
        start = time()
        ica = PCA(n_components=n)
        ica.fit_transform(train)
        t += time() - start
    f.write("%.3f\t%.3f\t%.3f\n" % (1, t/a , 0.0))


if __name__ == "__main__":
    # run_stats('../')
    # pca_cardio_scatter_plot('../')
    pca_loan_scatter_plot('../')
