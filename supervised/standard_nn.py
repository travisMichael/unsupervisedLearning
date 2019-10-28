from sklearn.neural_network import MLPClassifier
from utils import save_model, load_data, train_and_time
from utils import write_learning_curve_stats


def train_neural_net(path):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(3, 3), max_iter=1000, random_state=1)

    f = open("stats/nn_stats.txt","w+")

    write_learning_curve_stats(model_1, x_train, y_train, f, [10, 100, 500, 1000, 3000, 5000, 8000, 10000])

    f.close()


if __name__ == "__main__":

    train_neural_net('../')
