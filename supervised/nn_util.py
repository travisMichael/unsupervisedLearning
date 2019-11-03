from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from time import time
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def run_decomp_nn(k, estimator, f, x_train, x_test, y_train, y_test):
    ica_x_train = estimator.fit_transform(x_train)
    ica_x_test = estimator.transform(x_test)
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
    f.write('%3f\t%.4f\t%.3f\t%.3f\n' % (k,end, acc, 0.0))


def run_cluster_nn(k, estimator, f, x_train, x_test, y_train, y_test):
    print('running...')
    estimator.fit(x_train)
    predictions = estimator.predict(x_train)
    predictions = np.reshape(predictions, (-1, 1))
    test_predictions = estimator.predict(x_test)
    test_predictions = np.reshape(test_predictions, (-1, 1))

    enc = OneHotEncoder()
    enc.fit(predictions)
    train = enc.transform(predictions).toarray()
    test = enc.transform(test_predictions).toarray()

    # scaler = StandardScaler()
    # scaler.fit(predictions)
    # ica_x_train = scaler.transform(predictions)
    # ica_x_test = scaler.transform(test_predictions)
    start = time()
    model_1 = MLPClassifier(solver='sgd', learning_rate_init=0.001, validation_fraction=0.1, alpha=1e-6, hidden_layer_sizes=(5,5), max_iter=5000, random_state=1)
    model_1.fit(train, y_train)
    end = time() - start
    results = model_1.predict(test)
    acc = accuracy_score(y_test, results)
    f.write('%3f\t%.4f\t%.3f\t%.3f\n' % (k,end, acc, 0.0))
