from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
from utils import load_data
from sklearn.cluster import KMeans
from _heapq import heappush, heappop
import seaborn as sns
from scipy.stats import kurtosis


def pca_cardio_distribution_plot(path):

    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    ica = FastICA(n_components=4)
    ica.fit_transform(x_train)

    ica_x_train = ica.transform(x_train)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(x_train)
    print(kmeans.labels_)

    p = kmeans.predict(x_train)

    h = []
    for i in range(15):
        heappush(h, (-1 * len(p[p==i]), i))
        print(str(i) + " " + str(len(p[p==i])))

    index_1 = heappop(h)[1]
    index_2 = heappop(h)[1]
    index_3 = heappop(h)[1]

    dist_1 = ica_x_train[:, 0][p==index_1].ravel()
    dist_2 = ica_x_train[:, 0][p==index_2].ravel()
    dist_3 = ica_x_train[:, 0][p==index_3].ravel()

    sns.distplot(dist_1, hist=False, rug=True, color='red')
    sns.distplot(dist_2, hist=False, rug=True, color='blue')
    sns.distplot(dist_3, hist=False, rug=True, color='orange')

    plt.title('Cardio PCA Distribution')
    plt.savefig('plots/cardio_pca_distribution.png')

    print("")
    print(kurtosis(dist_1))
    print(kurtosis(dist_2))
    print(kurtosis(dist_3))


def pca_loan_distribution_plot(path):

    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    ica = FastICA(n_components=4)
    ica.fit_transform(x_train)

    ica_x_train = ica.transform(x_train)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(x_train)
    print(kmeans.labels_)

    p = kmeans.predict(x_train)

    h = []
    for i in range(15):
        heappush(h, (-1 * len(p[p==i]), i))
        print(str(i) + " " + str(len(p[p==i])))

    index_1 = heappop(h)[1]
    index_2 = heappop(h)[1]
    index_3 = heappop(h)[1]

    dist_1 = ica_x_train[:, 0][p==index_1].ravel()
    dist_2 = ica_x_train[:, 0][p==index_2].ravel()
    dist_3 = ica_x_train[:, 0][p==index_3].ravel()

    sns.distplot(dist_1, hist=False, rug=True, color='red')
    sns.distplot(dist_2, hist=False, rug=True, color='blue')
    sns.distplot(dist_3, hist=False, rug=True, color='orange')

    plt.title('Financial Loan PCA Distribution')
    plt.savefig('plots/loan_pca_distribution.png')

    print("")
    print(kurtosis(dist_1))
    print(kurtosis(dist_2))
    print(kurtosis(dist_3))


if __name__ == "__main__":
    pca_cardio_distribution_plot('../')
