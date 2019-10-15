from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from utils import save_model, load_data, train_and_time, save_figure


def run(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    X, y = load_data(path + 'data/' + data_set + '/train/')

    kmeans = KMeans(n_clusters=6, random_state=0).fit(x_train)
    print(kmeans.labels_)
    results = kmeans.predict(X)

    total = 0
    class_1 = 0
    for i in range(len(results)):
        total += 1
        if (results[i] == 0):
            class_1 += 1
        else:
            results[i] = 1

    print(accuracy_score(y, results))
    print(class_1, total)
    print(x_train.shape)


if __name__ == "__main__":
    # train_neural_net('../', False)
    run('../', "False")
