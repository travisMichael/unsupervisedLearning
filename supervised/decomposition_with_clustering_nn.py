from sklearn.neural_network import MLPClassifier
# from visualization_utils import multiple_learning_curves_plot
from utils import save_model, load_data, train_and_time
from sklearn.decomposition import PCA
from utils import write_learning_curve_stats
from sklearn.cluster import KMeans
import numpy as np


def train_neural_net_with_pca_k_means_loan_data(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    # x_train = x_train[0:1000,:]
    # y_train = y_train[0:1000]
    X, y = load_data(path + 'data/' + data_set + '/test/')

    pca = PCA(n_components=2)
    pca_x_train = pca.fit_transform(x_train)

    estimator = KMeans(n_clusters=3, random_state=0)

    estimator.fit(pca_x_train)
    results = estimator.predict(pca_x_train)
    results = np.reshape(results, (-1, 1))

    if with_plots == "False":
        model_1 = train_and_time(MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(3, 3), random_state=1), x_train, y_train)

    else:
        model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)

        f = open("pca_cluster_nn_stats.txt","w+")
        write_learning_curve_stats(model_1, results, y_train, f)
        f.close()


if __name__ == "__main__":
    # train_neural_net('../', False)
    # b = np.reshape(np.array([1, 2, 3]), (-1, 1))
    train_neural_net_with_pca_k_means_loan_data('../', "True")
