import matplotlib.pyplot as plt


def decomposition_runtime_plot_cardio(path):
    k, s = load_cluster_stats(path + "pca_cardio_runtime_stats.txt")
    k_2, s_2 = load_cluster_stats(path + "ica_cardio_runtime_stats.txt")
    k_3, s_3 = load_cluster_stats(path + "svd_cardio_runtime_stats.txt")
    k_4, s_4 = load_cluster_stats(path + "grp_cardio_runtime_stats.txt")

    plt = plot_multiple(
        [k, k_2, k_3, k_4],
        [s, s_2, s_3, s_4],
        ["black", "m", "g", "b"],
        ["PCA", "ICA", "SVD", "GRP"]
    )
    plt.xlabel('k')
    plt.ylabel('Runtime (s)')
    plt.title('Decomposition Runtime by Algorithm')
    plt.legend()
    # plt.show()
    plt.savefig("plots/decomposition_runtime_plot_cardio.png")


def decomposition_runtime_plot_loan(path):
    k, s = load_cluster_stats(path + "pca_loan_runtime_stats.txt")
    k_2, s_2 = load_cluster_stats(path + "ica_loan_runtime_stats.txt")
    k_3, s_3 = load_cluster_stats(path + "svd_loan_runtime_stats.txt")
    k_4, s_4 = load_cluster_stats(path + "grp_loan_runtime_stats.txt")
    k = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt = plot_multiple(
        [k, k, k, k],
        [s, s_2, s_3, s_4],
        ["black", "m", "g", "b"],
        ["PCA", "ICA", "SVD", "GRP"]
    )
    plt.xlabel('k')
    plt.ylabel('Runtime (s)')
    plt.title('Decomposition Runtime by Algorithm')
    plt.legend()
    # plt.show()
    plt.savefig("plots/decomposition_runtime_plot_loan.png")


def plot_single(x, y, color_list, label_list):
    plt.figure()
    # plt.legend(loc="best")
    plt.plot(x, y, '-o', color='black', label='dd')
    return plt


def plot_multiple(x, y, color_list, label_list):
    plt.figure()
    # plt.legend(loc="best")
    for i in range(len(x)):
        plt.plot(x[i], y[i], '-o', color=color_list[i], label=label_list[i])
    return plt


def load_cluster_stats(file):
    k = 1
    seconds = []
    k_list = []
    with open(file) as fp:
        line = fp.readline()
        while line:
            components = line.split("\t")
            k_list.append(k)
            seconds.append(float(components[1]))
            k += 1
            line = fp.readline()

    return k_list, seconds


if __name__ == "__main__":
    # decomposition_runtime_plot_cardio('stats/')
    decomposition_runtime_plot_loan('stats/')
