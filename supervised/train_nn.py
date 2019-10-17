# Resources
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html
from sklearn.neural_network import MLPClassifier
# from visualization_utils import multiple_learning_curves_plot
from utils import save_model, load_data, train_and_time, save_figure


def train_neural_net_with_loan_data(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if with_plots == "False":
        model_1 = train_and_time(MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(20, 5), random_state=1), x_train, y_train)
        model_2 = train_and_time(MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(50, 5), random_state=1), x_train, y_train)
        model_3 = train_and_time(MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(100, 5), random_state=1), x_train, y_train)
        model_4 = train_and_time(MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(500, 5), random_state=1), x_train, y_train)

        save_model(model_1, path + 'model/' + data_set, 'neural_net_model_1')
        save_model(model_2, path + 'model/' + data_set, 'neural_net_model_2')
        save_model(model_3, path + 'model/' + data_set, 'neural_net_model_3')
        save_model(model_4, path + 'model/' + data_set, 'neural_net_model_4')

    else:
        print('Training Neural Network...')
        model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-3, hidden_layer_sizes=(20, 5), random_state=1)
        model_2 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-3, hidden_layer_sizes=(50, 5), random_state=1)
        model_3 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-3, hidden_layer_sizes=(100, 5), random_state=1)
        model_4 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-3, hidden_layer_sizes=(500, 5), random_state=1)

        # plt = multiple_learning_curves_plot(
        #     [model_1, model_2, model_3, model_4],
        #     x_train, y_train,
        #     ["r", "y", "b", "m"],
        #     ['HLS = 20 x 5', 'HLS = 50 x 5', 'HLS = 100 x 5', 'HLS = 500 x 5']
        # )

        # plt.title("Neural Network with Varying Hidden Layer Size (HLS) \n Learning Curves")
        # plt.xlabel("Training examples")
        # plt.ylabel("F1 Score")
        # plt.grid()
        #
        # plt.legend(loc="best")
        # # plt.show()
        # save_figure(plt, path + "plot/" + data_set, 'neural_net_learning_curves.png')
        # print("done")


if __name__ == "__main__":
    # train_neural_net('../', False)
    train_neural_net_with_loan_data('../', "False")
