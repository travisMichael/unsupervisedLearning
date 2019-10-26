from sklearn.decomposition import FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import load_data
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from time import time
from _heapq import heappush, heappop


def ica_cardio_scatter_plot(path):

    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    ica = FastICA(n_components=4)
    ica.fit_transform(x_train)

    ica_x_train = ica.transform(x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(ica_x_train)
    print(kmeans.labels_)

    p = kmeans.predict(ica_x_train)

    for i in range(15):
        print(str(i) + " " + str(len(p[p==i])))

    plt.scatter(ica_x_train[:, 0][p==0].ravel(), ica_x_train[:,1][p==0].ravel(), alpha=.1, color='yellow')
    plt.scatter(ica_x_train[:, 0][p==1].ravel(), ica_x_train[:,1][p==1].ravel(), alpha=.1, color='orange')
    plt.scatter(ica_x_train[:, 0][p==7].ravel(), ica_x_train[:,1][p==7].ravel(), alpha=.1, color='blue')
    plt.scatter(ica_x_train[:, 0][p==9].ravel(), ica_x_train[:,1][p==9].ravel(), alpha=.1, color='red')
    plt.xlabel('ICA Component 1')
    plt.ylabel('ICA Component 2')
    plt.title('Cardiovascular Data Representation after ICA')
    # plt.show()
    plt.savefig("plots/ica_cardio.png")

    print(x_train.shape)


def ica_loan_scatter_plot(path):

    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    ica = FastICA(n_components=2)
    ica.fit_transform(x_train)

    ica_x_train = ica.transform(x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(ica_x_train)
    print(kmeans.labels_)

    p = kmeans.predict(ica_x_train)

    for i in range(15):
        print(str(i) + " " + str(len(p[p==i])))

    h = []
    for i in range(15):
        heappush(h, (-1 * len(p[p==i]), i))
        print(str(i) + " " + str(len(p[p==i])))

    index = heappop(h)[1]
    plt.scatter(ica_x_train[:, 0][p==index].ravel(), ica_x_train[:,1][p==index].ravel(), alpha=.1, color='red')
    index = heappop(h)[1]
    plt.scatter(ica_x_train[:, 0][p==index].ravel(), ica_x_train[:,1][p==index].ravel(), alpha=.1, color='orange')
    index = heappop(h)[1]
    plt.scatter(ica_x_train[:, 0][p==index].ravel(), ica_x_train[:,1][p==index].ravel(), alpha=.1, color='blue')

    plt.xlabel('ICA Component 1')
    plt.ylabel('ICA Component 2')
    plt.title('Financial Loan Data Representation after ICA')
    # plt.show()
    plt.savefig("plots/ica_loan.png")

    print(x_train.shape)


def run_stats(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    f = open("ica_runtime_stats.txt","w+")

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
    a = 10
    for i in range(a):
        start = time()
        ica = FastICA(n_components=n)
        ica.fit_transform(train)
        t += time() - start
    f.write("%.3f\t%.3f\t%.3f\n" % (1, t/a , 0.0))


if __name__ == "__main__":
    # run_stats('../')
    # ica_cardio_scatter_plot('../')
    ica_loan_scatter_plot('../')
