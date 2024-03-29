from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from utils import load_data
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from _heapq import heappush, heappop
from decomposition.decomp_util import time_estimator


# https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html
def grp_cardio_scatter_plot(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    grp = GaussianRandomProjection(n_components=3)
    grp_x_train = grp.fit_transform(x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(grp_x_train)
    print(kmeans.labels_)
    p = kmeans.predict(grp_x_train)

    p = kmeans.predict(grp_x_train)

    h = []
    for i in range(15):
        heappush(h, (-1 * len(p[p==i]), i))
        print(str(i) + " " + str(len(p[p==i])))

    index = heappop(h)[1]
    plt.scatter(grp_x_train[:, 0][p==index].ravel(), grp_x_train[:,1][p==index].ravel(), alpha=.1, color='yellow')
    index = heappop(h)[1]
    plt.scatter(grp_x_train[:, 0][p==index].ravel(), grp_x_train[:,1][p==index].ravel(), alpha=.1, color='orange')
    index = heappop(h)[1]
    plt.scatter(grp_x_train[:, 0][p==index].ravel(), grp_x_train[:,1][p==index].ravel(), alpha=.1, color='blue')
    index = heappop(h)[1]
    plt.scatter(grp_x_train[:, 0][p==index].ravel(), grp_x_train[:,1][p==index].ravel(), alpha=.1, color='red')
    plt.xlabel('GRP Component 1')
    plt.ylabel('GRP Component 2')
    plt.title('Cardiovascular Data Representation after GRP')
    plt.savefig("plots/grp_cardio.png")

    print()


def grp_loan_scatter_plot(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    grp = GaussianRandomProjection(n_components=2)
    grp_x_train = grp.fit_transform(x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(grp_x_train)
    print(kmeans.labels_)
    p = kmeans.predict(grp_x_train)

    p = kmeans.predict(grp_x_train)

    h = []
    for i in range(15):
        heappush(h, (-1 * len(p[p==i]), i))
        print(str(i) + " " + str(len(p[p==i])))

    index = heappop(h)[1]
    plt.scatter(grp_x_train[:, 0][p==index].ravel(), grp_x_train[:,1][p==index].ravel(), alpha=.1, color='red')
    index = heappop(h)[1]
    plt.scatter(grp_x_train[:, 0][p==index].ravel(), grp_x_train[:,1][p==index].ravel(), alpha=.1, color='orange')
    index = heappop(h)[1]
    plt.scatter(grp_x_train[:, 0][p==index].ravel(), grp_x_train[:,1][p==index].ravel(), alpha=.1, color='blue')
    index = heappop(h)[1]
    plt.xlabel('GRP Component 1')
    plt.ylabel('GRP Component 2')
    plt.title('Financial Loan Data Representation after GRP')
    plt.savefig("plots/grp_loan.png")

    print()


def grp_runtime_stats(path, data_set):
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    f = open("stats/grp_" + data_set + "_runtime_stats.txt","w+")

    time_estimator(GaussianRandomProjection(n_components=2), f, x_train)
    time_estimator(GaussianRandomProjection(n_components=10), f, x_train)
    time_estimator(GaussianRandomProjection(n_components=20), f, x_train)
    time_estimator(GaussianRandomProjection(n_components=30), f, x_train)
    time_estimator(GaussianRandomProjection(n_components=40), f, x_train)
    time_estimator(GaussianRandomProjection(n_components=50), f, x_train)
    time_estimator(GaussianRandomProjection(n_components=60), f, x_train)
    time_estimator(GaussianRandomProjection(n_components=70), f, x_train)
    time_estimator(GaussianRandomProjection(n_components=80), f, x_train)
    time_estimator(GaussianRandomProjection(n_components=90), f, x_train)
    time_estimator(GaussianRandomProjection(n_components=100), f, x_train)

    f.close()


if __name__ == "__main__":
    grp_runtime_stats('../', 'loan')
    # train_neural_net('../', False)
    # grp_cardio_scatter_plot('../')
    # grp_loan_scatter_plot('../')
