import matplotlib.pyplot as plt


def run(path):
    k, i, s = load_cluster_stats("cardiovascular_stats.txt")
    k_2, i_2, s_2 = load_cluster_stats("cardiovascular_pca_stats.txt")
    k_3, i_3, s_3 = load_cluster_stats("cardiovascular_ica_stats.txt")
    k_4, i_4, s_4 = load_cluster_stats("cardiovascular_random_projections_stats.txt")

    # plt = plot_multiple(
    #     [k, k_2],
    #     [i, i_2],
    #     ["r", "b", "m", "g"],
    #     ["model 1", "model 2", "model 3", "model 4"]
    # )
    plt = plot_multiple(
        [k, k_2, k_3, k_4],
        [s, s_2, s_3, s_4],
        ["r", "b", "m", "g"],
        ["model 1", "model 2", "model 3", "model 4"]
    )
    plt.xlabel('Problem Size')
    plt.ylabel('Optimal Value Achieved')
    plt.title('Optimal Values Achieved for SA Algorithms')
    plt.legend()
    plt.show()
    # plt.savefig("FlipFlop/SA_plot.png")
    print('hello')


def plot_single(x, y, color_list, label_list):
    plt.figure()
    # plt.legend(loc="best")
    plt.plot(x, y, '-o', color='black', label='dd')
    return plt


def plot_multiple(x, y, color_list, label_list):
    plt.figure()
    plt.legend(loc="best")
    for i in range(len(x)):
        plt.plot(x[i], y[i], '-o', color=color_list[i], label=label_list[i])
    return plt


def load_cluster_stats(file):
    print("Loading sa stats...")

    k = 1
    seconds = []
    inertia = []
    k_list = []
    with open(file) as fp:
        line = fp.readline()
        while line:
            components = line.split("\t")
            k_list.append(k)
            inertia.append(int(components[2]))
            seconds.append(float(components[1]))
            k += 1
            line = fp.readline()

    return k_list, inertia, seconds


if __name__ == "__main__":
    # train_neural_net('../', False)
    run('../')
