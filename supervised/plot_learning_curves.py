import matplotlib.pyplot as plt


def plot_loan_learning_curve(path):
    train_sizes, train_scores, test_scores = load_stats(path + "grp_nn_stats.txt")
    train_sizes_2, train_scores_2, test_scores_2 = load_stats(path + "pca_nn_stats.txt")
    train_sizes_3, train_scores_3, test_scores_3 = load_stats(path + "ica_nn_stats.txt")
    train_sizes_4, train_scores_4, test_scores_4 = load_stats(path + "svd_nn_stats.txt")

    plt = plot_multiple(
        [train_sizes, train_sizes_2, train_sizes_3, train_sizes_4],
        [train_scores, train_scores_2, train_scores_3, train_scores_4],
        [test_scores, test_scores_2, test_scores_3, test_scores_4],
        ["black", "m", "g", "b"],
        ["PCA", "ICA", "SVD", "GRP"]
    )
    plt.xlabel('Training Sample Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    ax = plt.gca()

    # Set x logaritmic
    ax.set_xscale('log')
    plt.legend()
    # plt.show()
    plt.savefig("plots/decomposition_learning_curve.png")


def plot_single(x, y, color_list, label_list):
    plt.figure()
    # plt.legend(loc="best")
    plt.plot(x, y, '-o', color='black', label='dd')
    return plt


def plot_multiple(train_size_list, train_score_list, test_score_list, color_list, label_list):
    plt.figure()
    for i in range(len(train_score_list)):
        plt.plot(train_size_list[i], train_score_list[i], '-o', color=color_list[i], label=label_list[i])
        plt.plot(train_size_list[i], test_score_list[i], '--', color=color_list[i], label=label_list[i])
    return plt


def load_stats(file):
    train_sizes = []
    train_scores = []
    test_scores = []
    with open(file) as fp:
        line = fp.readline()
        while line:
            components = line.split("\t")
            if len(components) > 2:
                train_sizes.append(int(components[0]))
                train_scores.append(float(components[1]))
                test_scores.append(float(components[2].split('\n')[0]))
            line = fp.readline()

    return train_sizes, train_scores, test_scores


if __name__ == "__main__":
    plot_loan_learning_curve('stats/')
