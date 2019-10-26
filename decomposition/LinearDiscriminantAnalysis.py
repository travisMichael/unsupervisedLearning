import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score


def run(path, with_plots):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    lda = LDA(n_components=5)
    lda_x_train = lda.fit_transform(x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(lda_x_train)
    print(kmeans.labels_)
    p = kmeans.predict(lda_x_train)

    plt.scatter(lda_x_train[:, 0][p==0].ravel(), lda_x_train[:,1][p==0].ravel(), alpha=.1, color='yellow')
    plt.scatter(lda_x_train[:, 0][p==6].ravel(), lda_x_train[:,1][p==6].ravel(), alpha=.1, color='orange')
    plt.scatter(lda_x_train[:, 0][p==2].ravel(), lda_x_train[:,1][p==2].ravel(), alpha=.1, color='blue')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()


if __name__ == "__main__":
    # train_neural_net('../', False)
    run('../', "False")
