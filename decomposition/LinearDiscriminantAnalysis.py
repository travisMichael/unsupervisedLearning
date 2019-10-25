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
    X, y = load_data(path + 'data/' + data_set + '/train/')

    pca = LDA(n_components=5)
    pca_x_train = pca.fit_transform(x_train)
    # Plot the explained variances
    features = range(pca.n_components_)
    # plt.bar(features, pca.explained_variance_ratio_, color='black')
    # plt.xlabel('PCA features')
    # plt.ylabel('variance %')
    # plt.xticks(features)
    # plt.show()

    # pca_x_train = pca.transform(x_train)
    # pca_x = pca.transform(X)


    kmeans = KMeans(n_clusters=10, random_state=0).fit(pca_x_train)
    print(kmeans.labels_)
    p = kmeans.predict(pca_x_train)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(x_train)
    print(kmeans.labels_)
    p_2 = kmeans.predict(x_train)

    for i in range(15):
        print(str(i) + " " + str(len(p[p==i])))

    print("%.6f" % homogeneity_score(p_2, p))




    print("")


    # print(class_1, total)
    # print(x_train.shape)


if __name__ == "__main__":
    # train_neural_net('../', False)
    run('../', "False")
