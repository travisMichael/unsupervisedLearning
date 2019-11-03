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

    print('ica...')
    f = open("stats/nn_ica_runtime_stats.txt","w+")
    run_decomp_nn(2, FastICA(n_components=2), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(10, FastICA(n_components=10), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(20, FastICA(n_components=20), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(30, FastICA(n_components=30), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(40, FastICA(n_components=40), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(50, FastICA(n_components=50), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(60, FastICA(n_components=60), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(70, FastICA(n_components=70), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(80, FastICA(n_components=80), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(90, FastICA(n_components=90), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(100, FastICA(n_components=100), f, x_train, x_test, y_train, y_test)

    print('pca...')
    f = open("stats/nn_pca_runtime_stats.txt","w+")
    run_decomp_nn(2, PCA(n_components=2), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(10, PCA(n_components=10), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(20, PCA(n_components=20), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(30, PCA(n_components=30), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(40, PCA(n_components=40), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(50, PCA(n_components=50), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(60, PCA(n_components=60), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(70, PCA(n_components=70), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(80, PCA(n_components=80), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(90, PCA(n_components=90), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(100, PCA(n_components=100), f, x_train, x_test, y_train, y_test)

    print('svd...')
    f = open("stats/nn_svd_runtime_stats.txt","w+")
    run_decomp_nn(2, TruncatedSVD(n_components=2), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(10, TruncatedSVD(n_components=10), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(20, TruncatedSVD(n_components=20), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(30, TruncatedSVD(n_components=30), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(40, TruncatedSVD(n_components=40), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(50, TruncatedSVD(n_components=50), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(60, TruncatedSVD(n_components=60), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(70, TruncatedSVD(n_components=70), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(80, TruncatedSVD(n_components=80), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(90, TruncatedSVD(n_components=90), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(100, TruncatedSVD(n_components=100), f, x_train, x_test, y_train, y_test)

    print('grp...')
    f = open("stats/nn_grp_runtime_stats.txt","w+")
    run_decomp_nn(2, GaussianRandomProjection(n_components=2), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(10, GaussianRandomProjection(n_components=10), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(20, GaussianRandomProjection(n_components=20), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(30, GaussianRandomProjection(n_components=30), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(40, GaussianRandomProjection(n_components=40), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(50, GaussianRandomProjection(n_components=50), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(60, GaussianRandomProjection(n_components=60), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(70, GaussianRandomProjection(n_components=70), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(80, GaussianRandomProjection(n_components=80), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(90, GaussianRandomProjection(n_components=90), f, x_train, x_test, y_train, y_test)
    run_decomp_nn(100, GaussianRandomProjection(n_components=100), f, x_train, x_test, y_train, y_test)


if __name__ == "__main__":

    generate_ica_neural_net_runtime_stats('../')
