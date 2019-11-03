from sklearn.neural_network import MLPClassifier
from utils import save_model, load_data, train_and_time
from utils import write_learning_curve_stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import accuracy_score
from time import time
import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_ica_neural_net_runtime_stats(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    x_test, y_test = load_data(path + 'data/' + data_set + '/train/')

    f = open("stats/nn_ica_runtime_stats.txt","w+")

    # start = time()
    # model_1 = MLPClassifier(solver='sgd', learning_rate_init=0.001, validation_fraction=0.1, alpha=1e-6, hidden_layer_sizes=(5, 5), max_iter=5000, random_state=1)
    # model_1.fit(x_train, y_train)
    # end = time() - start
    # results = model_1.predict(x_test)
    # acc = accuracy_score(y_test, results)
    # f.write('%.4f\t%.3f\t%.3f\n' % (end, acc, 0.0))

    ica = FastICA(n_components=100)
    ica_x_train = ica.fit_transform(x_train)
    ica_x_test = ica.transform(x_test)
    scaler = StandardScaler()
    scaler.fit(ica_x_train)
    ica_x_train = scaler.transform(ica_x_train)
    ica_x_test = scaler.transform(ica_x_test)
    start = time()
    model_1 = MLPClassifier(solver='sgd', learning_rate_init=0.001, validation_fraction=0.1, alpha=1e-6, hidden_layer_sizes=(5, 5), max_iter=5000, random_state=1)
    model_1.fit(ica_x_train, y_train)
    end = time() - start
    results = model_1.predict(ica_x_test)
    acc = accuracy_score(y_test, results)
    f.write('%.4f\t%.3f\t%.3f\n' % (end, acc, 0.0))


def generate_neural_net_runtime_stats(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    x_test, y_test = load_data(path + 'data/' + data_set + '/train/')

    f = open("stats/nn_runtime_stats.txt","w+")

    start = time()
    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)
    model_1.fit(x_train, y_train)
    end = time() - start
    results = model_1.predict(x_test)
    acc = accuracy_score(y_test, results)
    f.write('%.4f\t%.3f\t%.3f\n' % (end, acc, 0.0))

    estimator = KMeans(n_clusters=3, random_state=0)
    estimator.fit(x_train)
    predictions = estimator.predict(x_train)
    predictions = np.reshape(predictions, (-1, 1))
    test_predictions = estimator.predict(x_test)
    test_predictions = np.reshape(test_predictions, (-1, 1))
    start = time()
    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)
    model_1.fit(predictions, y_train)
    end = time() - start
    results = model_1.predict(test_predictions)
    acc = accuracy_score(y_test, results)
    f.write('%.4f\t%.3f\t%.3f\n' % (end, acc, 0.0))

    estimator = GaussianMixture(n_components=3, random_state=0)
    estimator.fit(x_train)
    predictions = estimator.predict(x_train)
    predictions = np.reshape(predictions, (-1, 1))
    test_predictions = estimator.predict(x_test)
    test_predictions = np.reshape(test_predictions, (-1, 1))
    start = time()
    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)
    model_1.fit(predictions, y_train)
    end = time() - start
    results = model_1.predict(test_predictions)
    acc = accuracy_score(y_test, results)
    f.write('%.4f\t%.3f\t%.3f\n' % (end, acc, 0.0))

    pca = PCA(n_components=2)
    pca_x_train = pca.fit_transform(x_train)
    pca_x_test = pca.transform(x_test)
    start = time()
    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)
    model_1.fit(pca_x_train, y_train)
    end = time() - start
    results = model_1.predict(pca_x_test)
    acc = accuracy_score(y_test, results)
    f.write('%.4f\t%.3f\t%.3f\n' % (end, acc, 0.0))

    ica = FastICA(n_components=2)
    ica_x_train = ica.fit_transform(x_train)
    ica_x_test = ica.transform(x_test)
    start = time()
    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)
    model_1.fit(ica_x_train, y_train)
    end = time() - start
    results = model_1.predict(ica_x_test)
    acc = accuracy_score(y_test, results)
    f.write('%.4f\t%.3f\t%.3f\n' % (end, acc, 0.0))

    svd = TruncatedSVD(n_components=2)
    svd_x_train = svd.fit_transform(x_train)
    svd_x_test = svd.transform(x_test)
    start = time()
    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)
    model_1.fit(svd_x_train, y_train)
    end = time() - start
    results = model_1.predict(svd_x_test)
    acc = accuracy_score(y_test, results)
    f.write('%.4f\t%.3f\t%.3f\n' % (end, acc, 0.0))

    grp = GaussianRandomProjection(n_components=2)
    grp_x_train = grp.fit_transform(x_train)
    grp_x_test = grp.transform(x_test)
    start = time()
    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)
    model_1.fit(grp_x_train, y_train)
    end = time() - start
    results = model_1.predict(grp_x_test)
    acc = accuracy_score(y_test, results)
    f.write('%.4f\t%.3f\t%.3f\n' % (end, acc, 0.0))

    f.close()


if __name__ == "__main__":

    generate_ica_neural_net_runtime_stats('../')
