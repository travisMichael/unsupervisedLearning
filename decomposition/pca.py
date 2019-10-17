import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
from sklearn.cluster import KMeans

def run(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    X, y = load_data(path + 'data/' + data_set + '/train/')

    pca = PCA(n_components=2)
    pca_x_train = pca.fit_transform(x_train)
    # Plot the explained variances
    features = range(pca.n_components_)
    # plt.bar(features, pca.explained_variance_ratio_, color='black')
    # plt.xlabel('PCA features')
    # plt.ylabel('variance %')
    # plt.xticks(features)
    # plt.show()

    # pca_x_train = pca.transform(x_train)
    pca_x = pca.transform(X)


    kmeans = KMeans(n_clusters=4, random_state=0).fit(pca_x_train)
    print(kmeans.labels_)
    results = kmeans.predict(pca_x)

    total = 0
    class_1 = 0
    for i in range(len(results)):
        total += 1
        if (results[i] == 0):
            class_1 += 1
        else:
            results[i] = 1

    print(accuracy_score(y, results))
    print(class_1, total)
    print("---")

    model = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(3, 3), random_state=1).fit(pca_x_train, y_train)
    #
    results = model.predict(pca_x)
    #
    print(accuracy_score(y, results))
    #
    model_2 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(3, 3), random_state=1).fit(x_train, y_train)

    results = model_2.predict(X)

    print(accuracy_score(y, results))

    # plt.scatter(principalComponents[0], principalComponents[1], alpha=.1, color='black')
    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')
    # plt.show()

    # print(class_1, total)
    # print(x_train.shape)


if __name__ == "__main__":
    # train_neural_net('../', False)
    run('../', "False")
