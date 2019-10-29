import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import mean_squared_error
from time import time
from _heapq import heappush, heappop
from decomposition.decomp_util import reconstruct, time_estimator
from sklearn.metrics.cluster import homogeneity_score
from sklearn.mixture import GaussianMixture


def calculate_cardio_kmeans_homogeneity_scores(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    x_test, y_test = load_data(path + 'data/' + data_set + '/test/')

    f = open("stats/pca_cardio_kmeans_homogeneity_scores.txt","w+")
    reduce_kmeans_and_score(1, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(2, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(3, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(4, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(5, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(6, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(7, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(8, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(9, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(10, x_train, y_train, x_test, y_test, f)
    f.close()


def calculate_cardio_EM_homogeneity_scores(path):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    x_test, y_test = load_data(path + 'data/' + data_set + '/test/')

    f = open("stats/pca_cardio_EM_homogeneity_scores.txt","w+")
    reduce_EM_and_score(1, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(2, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(3, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(4, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(5, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(6, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(7, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(8, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(9, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(10, x_train, y_train, x_test, y_test, f)
    f.close()


def calculate_loan_kmeans_homogeneity_scores(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    x_test, y_test = load_data(path + 'data/' + data_set + '/test/')

    f = open("stats/pca_loan_kmeans_homogeneity_scores.txt","w+")
    reduce_kmeans_and_score(1, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(2, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(3, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(4, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(5, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(6, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(7, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(8, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(9, x_train, y_train, x_test, y_test, f)
    reduce_kmeans_and_score(10, x_train, y_train, x_test, y_test, f)
    f.close()


def calculate_loan_EM_homogeneity_scores(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    x_test, y_test = load_data(path + 'data/' + data_set + '/test/')

    f = open("stats/pca_loan_EM_homogeneity_scores.txt","w+")
    reduce_EM_and_score(1, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(2, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(3, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(4, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(5, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(6, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(7, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(8, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(9, x_train, y_train, x_test, y_test, f)
    reduce_EM_and_score(10, x_train, y_train, x_test, y_test, f)
    f.close()


def reduce_kmeans_and_score(k, x_train, y_train, x_test, y_test, f):
    print('scoring...')
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x_train)
    base_predictions = kmeans.predict(x_test)
    pca = PCA(n_components=k)
    pca_x_train = pca.fit_transform(x_train)
    pca_x_test = pca.transform(x_test)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pca_x_train)
    predictions = kmeans.predict(pca_x_test)

    base_score = homogeneity_score(base_predictions, predictions)
    true_score = homogeneity_score(y_test, predictions)

    f.write('%.3f\t%.3f\t%.3f\n' % (base_score, true_score, 0.0))


def reduce_EM_and_score(k, x_train, y_train, x_test, y_test, f):
    print('scoring...')
    kmeans = GaussianMixture(n_components=k, random_state=0).fit(x_train)
    base_predictions = kmeans.predict(x_test)
    pca = PCA(n_components=k)
    pca_x_train = pca.fit_transform(x_train)
    pca_x_test = pca.transform(x_test)
    kmeans = GaussianMixture(n_components=k, random_state=0).fit(pca_x_train)
    predictions = kmeans.predict(pca_x_test)

    base_score = homogeneity_score(base_predictions, predictions)
    true_score = homogeneity_score(y_test, predictions)

    f.write('%.3f\t%.3f\t%.3f\n' % (base_score, true_score, 0.0))


if __name__ == "__main__":
    # calculate_cardio_kmeans_homogeneity_scores('../')
    # calculate_cardio_EM_homogeneity_scores('../')
    # calculate_loan_kmeans_homogeneity_scores('../')
    calculate_loan_EM_homogeneity_scores('../')