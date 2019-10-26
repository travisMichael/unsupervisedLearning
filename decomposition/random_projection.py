from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from utils import load_data
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


# https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html
def run(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    X, y = load_data(path + 'data/' + data_set + '/train/')

    grp = GaussianRandomProjection(n_components=2)
    grp_x_train = grp.fit_transform(x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(grp_x_train)
    print(kmeans.labels_)
    p = kmeans.predict(grp_x_train)

    plt.scatter(grp_x_train[:, 0][p==0].ravel(), grp_x_train[:,1][p==0].ravel(), alpha=.1, color='yellow')
    plt.scatter(grp_x_train[:, 0][p==6].ravel(), grp_x_train[:,1][p==6].ravel(), alpha=.1, color='orange')
    plt.scatter(grp_x_train[:, 0][p==2].ravel(), grp_x_train[:,1][p==2].ravel(), alpha=.1, color='blue')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()


if __name__ == "__main__":
    # train_neural_net('../', False)
    run('../', "False")
