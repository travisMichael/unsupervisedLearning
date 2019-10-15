import os
import time
import pickle
from sklearn.metrics import f1_score


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
