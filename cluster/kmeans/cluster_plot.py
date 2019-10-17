import matplotlib.pyplot as plt


def run(path):
    k, i, s = load_cluster_stats("loan_stats.txt")

    plt = plot_multiple(
        k,
        i,
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


def plot_multiple(x, y, color_list, label_list):
    plt.figure()
    # plt.legend(loc="best")
    plt.plot(x, y, '-o', color='black', label='dd')
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
