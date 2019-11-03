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
from supervised.nn_util import run_decomp_nn, run_cluster_nn


def generate_ica_neural_net_runtime_stats(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    x_test, y_test = load_data(path + 'data/' + data_set + '/train/')

    f = open("stats/nn_ica_runtime_stats.txt","w+")

    run_cluster_nn(2, GaussianMixture(n_components=90), f, x_train, x_test, y_train, y_test)
    # run_nn(10, FastICA(n_components=10), f, x_train, x_test, y_train, y_test)
    # run_nn(20, FastICA(n_components=20), f, x_train, x_test, y_train, y_test)
    # run_nn(30, FastICA(n_components=30), f, x_train, x_test, y_train, y_test)
    # run_nn(40, FastICA(n_components=40), f, x_train, x_test, y_train, y_test)
    # run_nn(50, FastICA(n_components=50), f, x_train, x_test, y_train, y_test)
    # run_nn(60, FastICA(n_components=60), f, x_train, x_test, y_train, y_test)
    # run_nn(70, FastICA(n_components=70), f, x_train, x_test, y_train, y_test)
    # run_nn(80, FastICA(n_components=80), f, x_train, x_test, y_train, y_test)
    # run_nn(90, FastICA(n_components=90), f, x_train, x_test, y_train, y_test)
    # run_nn(100, FastICA(n_components=100), f, x_train, x_test, y_train, y_test)



if __name__ == "__main__":

    generate_ica_neural_net_runtime_stats('../')
