import os
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import time


def save_model(model, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)

    pickle.dump(model, open(path + "/" + filename, 'wb'))


def save_figure(plt, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + "/" + filename)


def load_data(path):
    x_train_file = open(path + 'x', 'rb')
    y_train_file = open(path + 'y', 'rb')
    x_train = pickle.load(x_train_file)
    y_train = pickle.load(y_train_file)
    x_train_file.close()
    y_train_file.close()
    return x_train, y_train


def calculate_f1_score(model, X, y):
    start_time = time.time()
    predictions = model.predict(X)
    end_time = time.time() - start_time
    score = f1_score(y, predictions)
    print("f1 score", score, "prediction time", end_time)


def train_and_time(model, X, y):
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time() - start_time
    print("training time", end_time)
    return model


# np.linspace(.1, 1.0, 10)
def write_learning_curve_stats(model, x, y, f, sample_size_list):

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    print("Generating learning curve for model: ")
    start_time = time.time()
    if sample_size_list is None:
        sample_size_list = [5, 10, 15, 20, 50, 100, 10000]
    train_sizes, train_scores, test_scores = learning_curve(model, x, y, cv=cv, train_sizes=sample_size_list, scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    for i in range(len(train_sizes)):
        f.write('%i\t%.3f\t%.3f\n' % (train_sizes[i], train_scores_mean[i], test_scores_mean[i]))

    end_time = time.time() - start_time
    f.write(str(end_time))

    print("Learning curve finished for model: " + str(end_time))


# def multiple_learning_curves_plot(model_list, x, y, colors, training_labels):
#     number_of_models = len(model_list)
#     cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
#     plt.figure()
#     plt.legend(loc="best")
#
#     for i in range(number_of_models):
#         print("Generating learning curve for model: " + str(i))
#         start_time = time.time()
#         model = model_list[i]
#         curve_color = colors[i]
#         label = training_labels[i]
#
#         train_sizes, train_scores, test_scores = learning_curve(model, x, y, cv=cv, n_jobs=2, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1')
#
#         train_scores_mean = np.mean(train_scores, axis=1)
#         train_scores_std = np.std(train_scores, axis=1)
#         test_scores_mean = np.mean(test_scores, axis=1)
#         test_scores_std = np.std(test_scores, axis=1)
#
#         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color=curve_color)
#         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color=curve_color)
#         plt.plot(train_sizes, train_scores_mean, color=curve_color, label=label)
#         plt.plot(train_sizes, test_scores_mean, color=curve_color, linestyle='dashed')
#         end_time = time.time() - start_time
#         print("Learning curve finished for model: " + str(i) + " " + str(end_time))
#     return plt
