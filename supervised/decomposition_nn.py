from sklearn.neural_network import MLPClassifier
# from visualization_utils import multiple_learning_curves_plot
from utils import save_model, load_data, train_and_time
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from utils import write_learning_curve_stats
from sklearn.random_projection import GaussianRandomProjection


def train_neural_net_with_pca_loan_data(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = PCA(n_components=2)
    pca_x_train = pca.fit_transform(x_train)

    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)

    f = open("stats/pca_nn_stats.txt","w+")

    write_learning_curve_stats(model_1, pca_x_train, y_train, f, None)

    f.close()


def train_neural_net_with_ica_loan_data(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = FastICA(n_components=7)
    pca_x_train = pca.fit_transform(x_train)

    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)

    f = open("stats/ica_nn_stats.txt","w+")

    write_learning_curve_stats(model_1, pca_x_train, y_train, f, None)

    f.close()


def train_neural_net_with_grp_loan_data(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = GaussianRandomProjection(n_components=3)
    pca_x_train = pca.fit_transform(x_train)

    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)

    f = open("stats/grp_nn_stats.txt","w+")

    write_learning_curve_stats(model_1, pca_x_train, y_train, f, None)

    f.close()


def train_neural_net_with_svd_loan_data(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    pca = GaussianRandomProjection(n_components=3)
    pca_x_train = pca.fit_transform(x_train)

    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(5, 5), max_iter=1000, random_state=1)

    f = open("stats/svd_nn_stats.txt","w+")

    write_learning_curve_stats(model_1, pca_x_train, y_train, f, None)

    f.close()


if __name__ == "__main__":
    # train_neural_net_with_pca_loan_data('../')
    # train_neural_net_with_ica_loan_data('../')
    # train_neural_net_with_grp_loan_data('../')
    train_neural_net_with_ica_loan_data('../')
