from sklearn.neural_network import MLPClassifier
from utils import save_model, load_data, train_and_time
from utils import write_learning_curve_stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np


def train_neural_net_with_kmeans_loan_data(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    estimator = KMeans(n_clusters=3, random_state=0)
    estimator.fit(x_train)
    predictions = estimator.predict(x_train)
    predictions = np.reshape(predictions, (-1, 1))

    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)

    f = open("stats/k-means_nn_stats.txt","w+")

    write_learning_curve_stats(model_1, predictions, y_train, f, [10, 100, 500, 1000, 3000, 5000, 8000, 10000])

    f.close()


def train_neural_net_with_EM_loan_data(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    estimator = GaussianMixture(n_components=3, random_state=0)
    estimator.fit(x_train)
    predictions = estimator.predict(x_train)
    predictions = np.reshape(predictions, (-1, 1))

    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)

    f = open("stats/EM_nn_stats.txt","w+")

    write_learning_curve_stats(model_1, predictions, y_train, f, [10, 100, 500, 1000, 3000, 5000, 8000, 10000])

    f.close()


if __name__ == "__main__":
    # train_neural_net_with_kmeans_loan_data('../')
    train_neural_net_with_EM_loan_data('../')
