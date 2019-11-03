import matplotlib.pyplot as plt


def generate_cardio_elbow_plot(path):
    k, i, s = load_cluster_stats(path + "cardiovascular_stats.txt")

    plt = plot_multiple(
        [k],
        [i],
        ["black", "m", "g"],
        ["model 1", "model 2", "model 3", "model 4"]
    )
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('K Means - Cardiovascular Dataset')
    # plt.legend()
    # plt.show()
    plt.savefig("plots/cardio_elbow_plot.png")


def generate_loan_elbow_plot(path):
    k, i, s = load_cluster_stats("loan_stats.txt")

    plt = plot_multiple(
        [k],
        [i],
        ["black", "m", "g"],
        ["model 1", "model 2", "model 3", "model 4"]
    )
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('K Means - Financial Loan Dataset')
    # plt.legend()
    # plt.show()
    plt.savefig("plots/loan_elbow_plot.png")


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
    inertia = []
    k_list = []
    with open(file) as fp:
        line = fp.readline()
        while line:
            components = line.split("\t")
            k_list.append(int(components[0]))
            inertia.append(int(components[2]))
            seconds.append(float(components[1]))
            k += 1
            line = fp.readline()

    return k_list, inertia, seconds


if __name__ == "__main__":
    # train_neural_net('../', False)
    # generate_cardio_elbow_plot('../')
    generate_loan_elbow_plot('../')
