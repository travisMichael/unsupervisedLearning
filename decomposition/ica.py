from sklearn.decomposition import FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import load_data
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from time import time


def run_scatter_plot(path, with_plots):

    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    ica = FastICA(n_components=5)
    ica.fit_transform(x_train)

    ica_x_train = ica.transform(x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(ica_x_train)
    print(kmeans.labels_)
    p = kmeans.predict(ica_x_train)

    plt.scatter(ica_x_train[:, 0][p==0].ravel(), ica_x_train[:,1][p==0].ravel(), alpha=.1, color='yellow')
    plt.scatter(ica_x_train[:, 0][p==6].ravel(), ica_x_train[:,1][p==6].ravel(), alpha=.1, color='orange')
    plt.scatter(ica_x_train[:, 0][p==2].ravel(), ica_x_train[:,1][p==2].ravel(), alpha=.1, color='blue')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

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
    run_stats('../')
    # run_scatter_plot('../', "False")
