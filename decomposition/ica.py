from sklearn.decomposition import FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import load_data

from sklearn.datasets import load_digits


def run(path, with_plots):

    # digits = load_digits()
    # data = digits.data
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    X, y = load_data(path + 'data/' + data_set + '/train/')

    ica = FastICA(n_components=5)
    principalComponents = ica.fit_transform(x_train)
    # Plot the explained variances
    # features = range(ica.n_components_)
    # plt.bar(features, pca.explained_variance_ratio_, color='black')
    # plt.xlabel('PCA features')
    # plt.ylabel('variance %')
    # plt.xticks(features)
    # plt.show()

    pca_x_train = ica.transform(x_train)
    pca_x = ica.transform(X)

    model = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(20, 5), random_state=1).fit(pca_x_train, y_train)

    results = model.predict(pca_x)

    print(accuracy_score(y, results))

    model_2 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(20, 5), random_state=1).fit(x_train, y_train)

    # results = model_2.predict(X)
    #
    # print(accuracy_score(y, results))
    #
    # plt.scatter(principalComponents[0], principalComponents[1], alpha=.1, color='black')
    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')
    # plt.show()

    # print(class_1, total)
    print(x_train.shape)


if __name__ == "__main__":
    # train_neural_net('../', False)
    run('../', "False")
