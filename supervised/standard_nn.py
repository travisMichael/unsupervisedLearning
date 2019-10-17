from sklearn.neural_network import MLPClassifier
from utils import save_model, load_data, train_and_time
from utils import write_learning_curve_stats


def train_neural_net_with_loan_data(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    # x_train = x_train[0:1000,:]
    # y_train = y_train[0:1000]
    X, y = load_data(path + 'data/' + data_set + '/test/')

    if with_plots == "False":
        model_1 = train_and_time(MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(3, 3), random_state=1), x_train, y_train)

    else:
        model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)

        f = open("standard_nn_stats.txt","w+")
        write_learning_curve_stats(model_1, x_train, y_train, f)
        f.close()


if __name__ == "__main__":
    # train_neural_net('../', False)

    train_neural_net_with_loan_data('../', "True")
